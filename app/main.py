import logging
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from app.gemma import AudioResult, ConfigError, GemmaUnavailable, ask_gemma, image_to_audio
from app.history import history_store
from app.supabase_client import store_interaction

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"

# The Files API sniffs the type from the extension, so the temp file has to end
# in the right one. Only formats we have reason to trust are allowed through.
ALLOWED_IMAGE_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_QUESTION_CHARS = 2000

app = FastAPI(title="gemma-starter")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


# A plain `def` (not async) on purpose: FastAPI runs it in a worker thread, and
# ask_gemma blocks on network calls and sleeps between retries. In an `async def`
# that would stall every other request.
@app.post("/api/ask")
def ask(
    question: str = Form(...),
    image: Optional[UploadFile] = File(None),
) -> dict:
    question = question.strip()
    if not question:
        raise HTTPException(400, "Please type a question.")
    if len(question) > MAX_QUESTION_CHARS:
        raise HTTPException(400, f"Question is too long (max {MAX_QUESTION_CHARS} characters).")

    # Browsers send an empty file part when nothing was chosen, so check the
    # filename rather than just `is not None`.
    has_image = image is not None and bool(image.filename)

    try:
        if not has_image:
            answer = ask_gemma(question)
        else:
            suffix = ALLOWED_IMAGE_TYPES.get(image.content_type or "")
            if suffix is None:
                raise HTTPException(415, "Please upload a JPEG, PNG or WebP image.")
            data = image.file.read(MAX_IMAGE_BYTES + 1)
            if len(data) > MAX_IMAGE_BYTES:
                raise HTTPException(413, f"Image is too large (max {MAX_IMAGE_BYTES // 1024 // 1024} MB).")
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / f"upload{suffix}"
                path.write_bytes(data)
                answer = ask_gemma(question, path)
    except HTTPException:
        raise
    except GemmaUnavailable:
        raise HTTPException(503, "Gemma is busy right now. Please try again in a minute.")
    except ConfigError:
        log.exception("server misconfigured")
        raise HTTPException(500, "The server isn't configured correctly.")
    except Exception as exc:
        code = getattr(exc, "code", None)
        err_msg = str(exc)
        if code == 401 or "UNAUTHENTICATED" in err_msg or "API key not valid" in err_msg:
            log.warning("Authentication failure with Gemini API: %s", exc)
            raise HTTPException(
                401,
                "Invalid or unauthorized Gemini API key. Please check GEMINI_API_KEY in .env (get a valid key at https://aistudio.google.com/apikey).",
            )
        if code == 403 or "PERMISSION_DENIED" in err_msg:
            log.warning("Permission denied with Gemini API: %s", exc)
            raise HTTPException(
                403,
                "Permission denied or quota exceeded for GEMINI_API_KEY. Please verify your key permissions.",
            )
        # Non-transient API errors land here (bad key, rejected request). The
        # details go to the log, not to the browser.
        log.exception("Gemma request failed")
        raise HTTPException(502, "The model rejected this request.")

    # Record successfully completed interaction in history
    entry = history_store.add(
        kind="ask",
        prompt_or_query=question,
        result_summary=answer.text[:120] + ("..." if len(answer.text) > 120 else ""),
        vision_model=answer.model,
        attempts=answer.attempts,
        has_image=has_image,
    )
    # Sync with Supabase using the exact UUID
    store_interaction(
        kind="ask",
        prompt_or_query=question,
        result_summary=answer.text[:120] + ("..." if len(answer.text) > 120 else ""),
        entry_uuid=entry.id,
        vision_model=answer.model,
        attempts=answer.attempts,
        has_image=has_image,
    )

    return {"answer": answer.text, "model": answer.model, "attempts": answer.attempts, "id": entry.id}


# ---------------------------------------------------------------------------
# Image-to-Audio endpoint
# POST /api/soundscape   multipart: image file
# Returns: audio/wav bytes directly (stream it straight to the browser)
#
# The response also carries three custom headers so the frontend can display
# which models were used and what keyword was generated, without needing a
# separate metadata request.
# ---------------------------------------------------------------------------
@app.post("/api/soundscape")
def soundscape(
    image: Optional[UploadFile] = File(None),
    keyword: Optional[str] = Form(None),
) -> Response:
    """Accept an image upload or a keyword; return a WAV soundscape.

    If a specific keyword (e.g. an animal name) is supplied, audio is synthesized
    directly for that sound. Otherwise, the vision model extracts the acoustic
    texture from the image before synthesis.
    """
    if keyword and keyword.strip():
        # Strip newlines/control chars — HTTP header values must be single-line ASCII
        raw_kw = keyword.strip()
        kw = " ".join(raw_kw.splitlines()).strip()[:200]
        kw_safe = kw.encode("ascii", "replace").decode("ascii")
        try:
            audio_bytes, mime_type, audio_model, audio_attempts = generate_audio(kw)
            return Response(
                content=audio_bytes,
                media_type=mime_type,
                headers={
                    "X-Keyword": kw_safe,
                    "X-Vision-Model": "Gemma 4 Vision",
                    "X-Audio-Model": audio_model,
                    "X-Total-Attempts": str(audio_attempts),
                    "Access-Control-Expose-Headers": (
                        "X-Keyword, X-Vision-Model, X-Audio-Model, X-Total-Attempts"
                    ),
                },
            )
        except GemmaUnavailable as exc:
            last = getattr(exc, "last_error", None)
            last_code = getattr(last, "code", None)
            last_msg  = str(last or "")
            if last_code == 429 or "RESOURCE_EXHAUSTED" in last_msg or "quota" in last_msg.lower():
                raise HTTPException(
                    429,
                    "Free Tier audio quota reached. Try again tomorrow or upgrade your plan at https://aistudio.google.com.",
                )
            raise HTTPException(503, "All models are busy right now. Please try again in a minute.")
        except ConfigError:
            log.exception("server misconfigured")
            raise HTTPException(500, "The server isn't configured correctly.")
        except Exception as exc:
            code = getattr(exc, "code", None)
            err_msg = str(exc)
            if code == 401 or "UNAUTHENTICATED" in err_msg or "API key not valid" in err_msg:
                log.warning("Authentication failure in soundscape pipeline: %s", exc)
                raise HTTPException(
                    401,
                    "Invalid or unauthorized Gemini API key. Please check GEMINI_API_KEY in .env (get a valid key at https://aistudio.google.com/apikey).",
                )
            if code == 403 or "PERMISSION_DENIED" in err_msg:
                log.warning("Permission denied in soundscape pipeline: %s", exc)
                raise HTTPException(
                    403,
                    "Permission denied or quota exceeded for GEMINI_API_KEY. Please verify your key permissions.",
                )
            if code == 429 or "RESOURCE_EXHAUSTED" in err_msg or "quota" in err_msg.lower():
                log.warning("Quota exhausted in soundscape pipeline: %s", exc)
                raise HTTPException(
                    429,
                    "Free Tier audio quota reached. Try again tomorrow or upgrade your plan at https://aistudio.google.com.",
                )
            log.exception("sound generation failed")
            raise HTTPException(502, "The model rejected this request.")

    if image is None or not image.filename:
        raise HTTPException(400, "Please provide an image or a sound keyword.")

    suffix = ALLOWED_IMAGE_TYPES.get(image.content_type or "")
    if suffix is None:
        raise HTTPException(415, "Please upload a JPEG, PNG or WebP image.")

    data = image.file.read(MAX_IMAGE_BYTES + 1)
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(413, f"Image is too large (max {MAX_IMAGE_BYTES // 1024 // 1024} MB).")

    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"upload{suffix}"
            path.write_bytes(data)
            result: AudioResult = image_to_audio(path)
    except GemmaUnavailable as exc:
        last = getattr(exc, "last_error", None)
        last_code = getattr(last, "code", None)
        last_msg  = str(last or "")
        if last_code == 429 or "RESOURCE_EXHAUSTED" in last_msg or "quota" in last_msg.lower():
            raise HTTPException(
                429,
                "Free Tier audio quota reached. Try again tomorrow or upgrade your plan at https://aistudio.google.com.",
            )
        raise HTTPException(503, "All models are busy right now. Please try again in a minute.")
    except ConfigError:
        log.exception("server misconfigured")
        raise HTTPException(500, "The server isn't configured correctly.")
    except Exception as exc:
        code = getattr(exc, "code", None)
        err_msg = str(exc)
        if code == 401 or "UNAUTHENTICATED" in err_msg or "API key not valid" in err_msg:
            log.warning("Authentication failure in soundscape pipeline: %s", exc)
            raise HTTPException(
                401,
                "Invalid or unauthorized Gemini API key. Please check GEMINI_API_KEY in .env (get a valid key at https://aistudio.google.com/apikey).",
            )
        if code == 403 or "PERMISSION_DENIED" in err_msg:
            log.warning("Permission denied in soundscape pipeline: %s", exc)
            raise HTTPException(
                403,
                "Permission denied or quota exceeded for GEMINI_API_KEY. Please verify your key permissions.",
            )
        log.exception("soundscape pipeline failed")
        raise HTTPException(502, "The model rejected this request.")

    # Record successfully generated soundscape in history
    entry = history_store.add(
        kind="soundscape",
        prompt_or_query=image.filename or "uploaded image",
        result_summary=result.keyword,
        vision_model=result.vision_model,
        audio_model=result.audio_model,
        attempts=result.total_attempts,
        has_image=True,
    )
    # Sync with Supabase using the exact UUID
    store_interaction(
        kind="soundscape",
        prompt_or_query=image.filename or "uploaded image",
        result_summary=result.keyword,
        entry_uuid=entry.id,
        vision_model=result.vision_model,
        audio_model=result.audio_model,
        attempts=result.total_attempts,
        has_image=True,
    )

    return Response(
        content=result.audio_bytes,
        media_type=result.mime_type,
        headers={
            "X-Entry-Id": entry.id,
            "X-Keyword": result.keyword,
            "X-Vision-Model": result.vision_model,
            "X-Audio-Model": result.audio_model,
            "X-Total-Attempts": str(result.total_attempts),
            # Allow the browser to read these custom headers via JS fetch.
            "Access-Control-Expose-Headers": (
                "X-Entry-Id, X-Keyword, X-Vision-Model, X-Audio-Model, X-Total-Attempts"
            ),
        },
    )


# ---------------------------------------------------------------------------
# History / Latest Changes endpoint
# GET /api/history
# ---------------------------------------------------------------------------
@app.get("/api/history")
def get_history(limit: int = 20) -> dict:
    """Return the recent activity log (Q&A and soundscapes generated)."""
    return {"history": history_store.list_entries(limit=limit)}


@app.delete("/api/history")
def clear_history() -> dict:
    """Clear in-memory activity log."""
    history_store.clear()
    return {"status": "cleared"}



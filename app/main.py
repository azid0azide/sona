import logging
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from app.gemma import AudioResult, ConfigError, GemmaUnavailable, ask_gemma, image_to_audio

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
    except Exception:
        # Non-transient API errors land here (bad key, rejected request). The
        # details go to the log, not to the browser.
        log.exception("Gemma request failed")
        raise HTTPException(502, "The model rejected this request.")

    return {"answer": answer.text, "model": answer.model, "attempts": answer.attempts}


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
    image: UploadFile = File(...),
) -> Response:
    """Accept an image upload; return a WAV soundscape that matches it.

    Pipeline
    --------
    1. Gemma 4 (vision) reads the image → one-line audio keyword
    2. Flash TTS → WAV bytes  (falls back to muse-glimmer if unavailable)
    """
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
    except GemmaUnavailable:
        raise HTTPException(503, "All models are busy right now. Please try again in a minute.")
    except ConfigError:
        log.exception("server misconfigured")
        raise HTTPException(500, "The server isn't configured correctly.")
    except Exception:
        log.exception("soundscape pipeline failed")
        raise HTTPException(502, "The model rejected this request.")

    return Response(
        content=result.audio_bytes,
        media_type=result.mime_type,
        headers={
            "X-Keyword": result.keyword,
            "X-Vision-Model": result.vision_model,
            "X-Audio-Model": result.audio_model,
            "X-Total-Attempts": str(result.total_attempts),
            # Allow the browser to read these custom headers via JS fetch.
            "Access-Control-Expose-Headers": (
                "X-Keyword, X-Vision-Model, X-Audio-Model, X-Total-Attempts"
            ),
        },
    )


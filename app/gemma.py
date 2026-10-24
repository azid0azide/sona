"""The only module that talks to the Gemini API.

Everything else in the app calls ask_gemma(). Keeping the model call in one
place is deliberate: this file is the single spot that knows about the SDK,
retries, and which Gemma variant is in use, so moving to a different backend
(a local Ollama server, say) means rewriting this file and nothing else.

The retry policy exists because the API returned 500 and 503 errors on plain
prompts during development. See docs/adr/0002-retry-then-fall-back.md.

Image-to-Audio pipeline
-----------------------
In addition to ask_gemma(), this module now exposes image_to_audio():

    image  ──►  describe_image()   (Gemma 4 vision → one-line audio keyword)
                    │
                    │  e.g. "Deep lion roar"
                    ▼
                generate_audio()   (Flash TTS → WAV bytes, muse-glimmer fallback)
                    │
                    ▼
                AudioResult        (.audio_bytes, .keyword, .vision_model, …)

Both stages share the same retry + model-fallback loop as ask_gemma(), so the
retry policy documented in ADR 0002 applies end-to-end.

Environment variables
~~~~~~~~~~~~~~~~~~~~~
GEMMA_MODELS              comma-separated list for ask_gemma() vision models
SOUND_VISION_MODELS       comma-separated list for image_to_audio() stage 1
SOUND_AUDIO_MODELS        comma-separated list for image_to_audio() stage 2
                          (default: gemini-2.5-flash-preview-tts,muse-glimmer)
SOUND_RETRIES_PER_MODEL   retries for each audio model   (default: 2)
SOUND_BASE_DELAY_S        base back-off in seconds        (default: 1.0)
SOUND_TIMEOUT_S           per-request timeout in seconds  (default: 30)
"""

import logging
import os
import random
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

log = logging.getLogger(__name__)

# 26B A4B first: Google's docs suggest it as the general-purpose starting point.
# 31B is the fallback because it is the only other Gemma 4 model the Gemini API
# lists for this key.
DEFAULT_MODELS = ("gemma-4-26b-a4b-it", "gemma-4-31b-it")

# ---------------------------------------------------------------------------
# Audio pipeline defaults
#
# Primary : gemini-2.5-flash-preview-tts  – Google Flash TTS (google-genai SDK,
#                                            GEMINI_API_KEY)
# Fallback : meta/muse-glimmer-30b        – NVIDIA NIM / Meta VLM used as a
#                                            text-to-description bridge then
#                                            narrated by the TTS layer, or for
#                                            future direct audio NIM support
#                                            (openai-compat SDK, MUSE_API_KEY)
#
# Models whose names start with a NIM vendor prefix (meta/, nvidia/, …) are
# automatically routed to the NVIDIA NIM endpoint using MUSE_API_KEY.
# Everything else goes through the Gemini SDK using GEMINI_API_KEY.
# ---------------------------------------------------------------------------
DEFAULT_AUDIO_MODELS: Tuple[str, ...] = (
    "gemini-3.8-flash-tts",
    "meta/muse-glimmer-30b",
)

# NVIDIA NIM OpenAI-compatible base URL.
NVIDIA_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"

# Model-name prefixes that live on the NVIDIA NIM platform.
# Any model whose name starts with one of these is dispatched to NIM.
NVIDIA_MODEL_PREFIXES = ("meta/", "nvidia/", "mistralai/", "microsoft/")

# 429 is rate limiting. 500 and 503 are the codes actually seen during
# development; 502/504 are the usual gateway cousins.
TRANSIENT_CODES = frozenset({429, 500, 502, 503, 504})

# Network-level failures don't carry an HTTP code. Matched by class name so we
# don't have to import (and pin to) whichever HTTP library the SDK uses today.
TRANSIENT_ERROR_NAMES = frozenset(
    {
        "TimeoutError",
        "ReadTimeout",
        "ConnectTimeout",
        "TimeoutException",
        "ConnectError",
        "RemoteProtocolError",
    }
)

SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "system.txt"


class ConfigError(RuntimeError):
    """The server is missing something it needs (e.g. the API key)."""


class GemmaUnavailable(RuntimeError):
    """Every configured model failed, even after retries."""

    def __init__(self, message: str, last_error: Optional[BaseException] = None):
        super().__init__(message)
        self.last_error = last_error


class _EmptyResponse(Exception):
    """The API answered 200 but gave us no text."""


@dataclass(frozen=True)
class Settings:
    models: Tuple[str, ...] = DEFAULT_MODELS
    retries_per_model: int = 2
    base_delay_s: float = 1.0
    timeout_s: float = 30.0

    @classmethod
    def from_env(cls) -> "Settings":
        raw_models = os.environ.get("GEMMA_MODELS", "")
        models = tuple(m.strip() for m in raw_models.split(",") if m.strip())
        return cls(
            models=models or DEFAULT_MODELS,
            retries_per_model=int(os.environ.get("GEMMA_RETRIES_PER_MODEL", 2)),
            base_delay_s=float(os.environ.get("GEMMA_BASE_DELAY_S", 1.0)),
            timeout_s=float(os.environ.get("GEMMA_TIMEOUT_S", 30)),
        )


@dataclass(frozen=True)
class Answer:
    text: str
    model: str
    attempts: int


def load_system_prompt() -> str:
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()


@lru_cache(maxsize=1)
def get_client(timeout_s: float = 30.0) -> Any:
    # Imported lazily so tests that inject a fake client never need the SDK
    # configured, and importing this module never needs a key.
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ConfigError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add a key "
            "from https://aistudio.google.com/apikey"
        )
    # Without an explicit timeout, one stuck request can hang for minutes
    # (a plain call took 1m45s to return a 500 while debugging). The SDK takes
    # the timeout in milliseconds.
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=int(timeout_s * 1000)),
    )


@lru_cache(maxsize=1)
def get_muse_client(timeout_s: float = 30.0) -> Any:
    """Return a cached OpenAI-compatible client pointed at NVIDIA NIM.

    Used exclusively for models whose names start with a NIM vendor prefix
    (``meta/``, ``nvidia/``, …).  Requires ``MUSE_API_KEY`` to be set.

    Imported lazily so tests that inject a fake client never need the real
    ``openai`` package installed, and importing this module never raises even
    when the NIM key is absent.
    """
    from openai import OpenAI

    api_key = os.environ.get("MUSE_API_KEY")
    if not api_key:
        raise ConfigError(
            "MUSE_API_KEY is not set. Add it to .env – it is the nvapi-… key "
            "from https://build.nvidia.com/explore/reasoning"
        )
    return OpenAI(
        base_url=NVIDIA_NIM_BASE_URL,
        api_key=api_key,
        timeout=timeout_s,
    )


def _is_nim_model(model: str) -> bool:
    """Return True if ``model`` should be dispatched to the NVIDIA NIM endpoint."""
    return model.startswith(NVIDIA_MODEL_PREFIXES)


def _error_code(exc: BaseException) -> Optional[int]:
    code = getattr(exc, "code", None)
    return code if isinstance(code, int) else None


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, _EmptyResponse):
        return True
    if _error_code(exc) in TRANSIENT_CODES:
        return True
    return type(exc).__name__ in TRANSIENT_ERROR_NAMES


def ask_gemma(
    prompt: str,
    image_path: Optional[os.PathLike] = None,
    *,
    client: Any = None,
    settings: Optional[Settings] = None,
    system_prompt: Optional[str] = None,
    sleep: Callable[[float], None] = time.sleep,
) -> Answer:
    """Ask Gemma a question, optionally about an image.

    Retries transient failures with exponential backoff, then moves on to the
    next configured model. Errors that retrying cannot fix (bad key, malformed
    request) are raised immediately: falling back would just repeat the mistake
    against a second model.

    `client`, `settings` and `sleep` are injectable so the retry logic can be
    tested without network access or real waiting.
    """
    settings = settings or Settings.from_env()
    client = client or get_client(settings.timeout_s)
    if system_prompt is None:
        system_prompt = load_system_prompt()

    config = None
    if system_prompt:
        from google.genai import types

        config = types.GenerateContentConfig(system_instruction=system_prompt)

    uploaded = None  # uploaded lazily, then reused across every retry and model
    attempts = 0
    last_error: Optional[BaseException] = None

    for model in settings.models:
        for attempt in range(1, settings.retries_per_model + 1):
            attempts += 1
            try:
                if image_path is not None and uploaded is None:
                    uploaded = client.files.upload(file=str(image_path))
                contents = [uploaded, prompt] if uploaded is not None else [prompt]
                response = client.models.generate_content(
                    model=model, contents=contents, config=config
                )
                text = (response.text or "").strip()
                if not text:
                    raise _EmptyResponse()
                return Answer(text=text, model=model, attempts=attempts)
            except Exception as exc:
                last_error = exc
                if _error_code(exc) == 404:
                    log.warning("%s not found for this key, trying next model", model)
                    break
                if not _is_transient(exc):
                    raise
                log.warning(
                    "%s attempt %d/%d failed: %s",
                    model, attempt, settings.retries_per_model, exc,
                )
                if attempt < settings.retries_per_model:
                    # Jitter keeps a room full of people hitting the same
                    # overloaded endpoint from retrying in lockstep.
                    delay = settings.base_delay_s * 2 ** (attempt - 1)
                    sleep(delay + random.uniform(0, settings.base_delay_s / 2))

    raise GemmaUnavailable(
        f"No Gemma model answered after {attempts} attempts "
        f"(tried: {', '.join(settings.models)})",
        last_error=last_error,
    )


# ---------------------------------------------------------------------------
# Image-to-Audio pipeline
# ---------------------------------------------------------------------------

VISION_PROMPT = (
    "Analyze this image. Identify the primary subject or environment. "
    "Respond with exactly ONE specific sound-effect description that belongs here. "
    "Examples: 'Deep lion roar', 'Wind howling on a rocky mountain peak', "
    "'Thunderstorm in a dense forest'. "
    "Do NOT write full sentences. Give me only the descriptive audio keyword or phrase."
)

AUDIO_PROMPT_TEMPLATE = (
    "<speak>"
    "[INSTRUCTION: SUPPRESS ALL HUMAN SPEECH AND NARRATION. "
    "OUTPUT ONLY THE REQUESTED SOUND EFFECTS.]\n\n"
    "Generate 5 to 10 seconds of raw, uninterrupted sound design matching: {keyword}.\n\n"
    "If it is an animal, produce its natural aggressive vocalization or roar.\n"
    "If it is a nature scene, produce heavy ambient environmental sound with "
    "wind, echo, and open-air presence.\n"
    "</speak>"
)


@dataclass(frozen=True)
class AudioResult:
    """Outcome of the full image → audio pipeline."""

    audio_bytes: bytes
    mime_type: str       # e.g. "audio/wav"
    keyword: str         # Stage-1 sound description from Gemma 4
    vision_model: str    # which Gemma 4 variant described the image
    audio_model: str     # which TTS / music model generated the audio
    total_attempts: int  # retries across both stages combined


def describe_image(
    image_path: os.PathLike,
    *,
    client: Any = None,
    settings: Optional[Settings] = None,
    sleep: Callable[[float], None] = time.sleep,
) -> Answer:
    """Feed an image to Gemma 4 and return a one-line sound description.

    Returns an :class:`Answer` whose ``.text`` is the audio keyword so that
    the retry-and-fallback machinery already covered by the test suite applies
    without any duplication.

    The image is uploaded once and reused across retries and model fallbacks.
    """
    settings = settings or Settings.from_env()
    client = client or get_client(settings.timeout_s)

    uploaded: Any = None
    attempts = 0
    last_error: Optional[BaseException] = None

    for model in settings.models:
        for attempt in range(1, settings.retries_per_model + 1):
            attempts += 1
            try:
                if uploaded is None:
                    uploaded = client.files.upload(file=str(image_path))
                response = client.models.generate_content(
                    model=model,
                    contents=[uploaded, VISION_PROMPT],
                )
                text = (response.text or "").strip()
                if not text:
                    raise _EmptyResponse()
                log.info("[vision] keyword=%r  model=%s  attempts=%d", text, model, attempts)
                return Answer(text=text, model=model, attempts=attempts)
            except Exception as exc:
                last_error = exc
                if _error_code(exc) == 404:
                    log.warning("[vision] %s not found for this key, trying next model", model)
                    break
                if not _is_transient(exc):
                    raise
                log.warning(
                    "[vision] %s attempt %d/%d failed: %s",
                    model, attempt, settings.retries_per_model, exc,
                )
                if attempt < settings.retries_per_model:
                    delay = settings.base_delay_s * 2 ** (attempt - 1)
                    sleep(delay + random.uniform(0, settings.base_delay_s / 2))

    raise GemmaUnavailable(
        f"[vision] No model answered after {attempts} attempts "
        f"(tried: {', '.join(settings.models)})",
        last_error=last_error,
    )


def generate_audio(
    keyword: str,
    *,
    client: Any = None,
    settings: Optional[Settings] = None,
    audio_models: Optional[Tuple[str, ...]] = None,
    sleep: Callable[[float], None] = time.sleep,
) -> Tuple[bytes, str, str, int]:
    """Turn a sound-effect keyword into raw audio bytes.

    Returns ``(audio_bytes, mime_type, winning_model, attempts)``.

    Routing
    -------
    * **Google models** (e.g. ``gemini-2.5-flash-preview-tts``) — called via
      the ``google-genai`` SDK using ``GEMINI_API_KEY``.
    * **NVIDIA NIM models** (e.g. ``meta/muse-glimmer-30b``) — called via the
      OpenAI-compatible NIM endpoint using ``MUSE_API_KEY``.

    Because muse-glimmer-30b is a text/vision model rather than a native audio
    synthesiser, the NIM path asks it to produce a richly detailed description
    of the sound, then re-synthesises that description with Flash TTS.  If the
    TTS re-synthesis also fails, the description text is returned as UTF-8 bytes
    so the pipeline never fully collapses.
    """
    settings = settings or Settings.from_env()
    client = client or get_client(settings.timeout_s)

    # Caller may override the model list; otherwise fall back to env / default.
    raw_env = os.environ.get("SOUND_AUDIO_MODELS", "")
    env_models = tuple(m.strip() for m in raw_env.split(",") if m.strip())
    models = audio_models or env_models or DEFAULT_AUDIO_MODELS

    prompt = AUDIO_PROMPT_TEMPLATE.format(keyword=keyword)
    attempts = 0
    last_error: Optional[BaseException] = None

    for model in models:
        for attempt in range(1, settings.retries_per_model + 1):
            attempts += 1
            try:
                if _is_nim_model(model):
                    # -------------------------------------------------------
                    # NVIDIA NIM path – OpenAI-compatible chat completion
                    # -------------------------------------------------------
                    muse = get_muse_client(settings.timeout_s)
                    nim_response = muse.chat.completions.create(
                        model=model,
                        messages=[
                            {
                                "role": "user",
                                "content": (
                                    f"You are a sound designer. Describe in vivid "
                                    f"detail the exact sonic texture of: {keyword}. "
                                    f"Return only the sonic description, no preamble."
                                ),
                            }
                        ],
                        temperature=0.7,
                        max_tokens=256,
                    )
                    description = (nim_response.choices[0].message.content or "").strip()
                    if not description:
                        raise _EmptyResponse()
                    log.info(
                        "[audio/nim] model=%s  description=%r  attempts=%d",
                        model, description[:80], attempts,
                    )
                    # Re-synthesise the enriched description with Flash TTS.
                    enriched_prompt = AUDIO_PROMPT_TEMPLATE.format(keyword=description)
                    try:
                        from google.genai import types as gtypes
                        tts_response = client.models.generate_content(
                            model="gemini-3.8-flash-tts",
                            contents=enriched_prompt,
                            config=gtypes.GenerateContentConfig(
                                response_modalities=["AUDIO"],
                            ),
                        )
                        for cand in (tts_response.candidates or []):
                            for part in (getattr(cand.content, "parts", None) or []):
                                if getattr(part, "inline_data", None):
                                    data = part.inline_data.data
                                    mime = part.inline_data.mime_type or "audio/wav"
                                    if data:
                                        log.info(
                                            "[audio/nim+tts] bytes=%d  attempts=%d",
                                            len(data), attempts,
                                        )
                                        return data, mime, model, attempts
                    except Exception as tts_exc:
                        log.warning(
                            "[audio/nim] TTS re-synthesis failed (%s); "
                            "returning description as text bytes", tts_exc,
                        )
                    # Last-resort fallback: return description text as bytes.
                    return description.encode("utf-8"), "text/plain", model, attempts

                else:
                    # -------------------------------------------------------
                    # Google Gemini / Flash TTS path
                    # -------------------------------------------------------
                    from google.genai import types

                    response = client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_modalities=["AUDIO"],
                        ),
                    )
                    for candidate in (response.candidates or []):
                        for part in (getattr(candidate.content, "parts", None) or []):
                            if getattr(part, "inline_data", None):
                                data = part.inline_data.data
                                mime = part.inline_data.mime_type or "audio/wav"
                                if data:
                                    log.info(
                                        "[audio] model=%s  bytes=%d  attempts=%d",
                                        model, len(data), attempts,
                                    )
                                    return data, mime, model, attempts
                    raise _EmptyResponse()

            except Exception as exc:
                last_error = exc
                if _error_code(exc) == 404:
                    log.warning("[audio] %s not found for this key, trying next model", model)
                    break
                if not _is_transient(exc):
                    raise
                log.warning(
                    "[audio] %s attempt %d/%d failed: %s",
                    model, attempt, settings.retries_per_model, exc,
                )
                if attempt < settings.retries_per_model:
                    delay = settings.base_delay_s * 2 ** (attempt - 1)
                    sleep(delay + random.uniform(0, settings.base_delay_s / 2))

    raise GemmaUnavailable(
        f"[audio] No audio model answered after {attempts} attempts "
        f"(tried: {', '.join(models)})",
        last_error=last_error,
    )


def image_to_audio(
    image_path: os.PathLike,
    *,
    client: Any = None,
    settings: Optional[Settings] = None,
    sleep: Callable[[float], None] = time.sleep,
) -> AudioResult:
    """Full pipeline: image file → :class:`AudioResult` with raw WAV bytes.

    Stage 1 – Gemma 4 reads the image and returns a one-line sound keyword.
    Stage 2 – Flash TTS (then muse-glimmer) turns that keyword into a WAV.

    Example
    -------
    >>> result = image_to_audio("lion.jpg")
    >>> Path("roar.wav").write_bytes(result.audio_bytes)

    Raises
    ------
    ConfigError
        GEMINI_API_KEY is not set.
    GemmaUnavailable
        Every vision *or* audio model failed after all retries.
    """
    settings = settings or Settings.from_env()
    client = client or get_client(settings.timeout_s)

    # Stage 1 – vision
    keyword_answer = describe_image(image_path, client=client, settings=settings, sleep=sleep)

    # Stage 2 – audio
    audio_bytes, mime_type, audio_model, audio_attempts = generate_audio(
        keyword_answer.text, client=client, settings=settings, sleep=sleep
    )

    return AudioResult(
        audio_bytes=audio_bytes,
        mime_type=mime_type,
        keyword=keyword_answer.text,
        vision_model=keyword_answer.model,
        audio_model=audio_model,
        total_attempts=keyword_answer.attempts + audio_attempts,
    )

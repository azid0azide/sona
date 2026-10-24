"""Tests for the retry and fallback policy in app/gemma.py.

A scripted fake client stands in for the Gemini API, so these run offline and
without sleeping. They pin down the behaviour that matters on stage: transient
errors are retried, an unhealthy model is abandoned for the next one, and
errors that retrying can't fix surface immediately.

The second section covers the image-to-audio pipeline (describe_image,
generate_audio, image_to_audio) using the same FakeClient infrastructure.
"""

import pytest

from app.gemma import (
    Answer,
    AudioResult,
    GemmaUnavailable,
    Settings,
    ask_gemma,
    describe_image,
    generate_audio,
    image_to_audio,
    load_system_prompt,
)


class FakeAPIError(Exception):
    """Mimics the SDK's errors, which expose the HTTP status as `.code`."""

    def __init__(self, code):
        super().__init__(f"HTTP {code}")
        self.code = code


class FakeResponse:
    def __init__(self, text):
        self.text = text


class FakePart:
    """Mimics a response part that contains inline audio data."""

    def __init__(self, data: bytes, mime_type: str = "audio/wav"):
        self.inline_data = type(
            "InlineData", (), {"data": data, "mime_type": mime_type}
        )()


class FakeCandidate:
    def __init__(self, parts):
        self.content = type("Content", (), {"parts": parts})()


class FakeAudioResponse:
    """Mimics a generate_content response that carries audio inline data."""

    def __init__(self, data: bytes, mime_type: str = "audio/wav"):
        self.text = None
        self.candidates = [FakeCandidate([FakePart(data, mime_type)])]


class FakeClient:
    """Plays back one scripted outcome per generate_content call."""

    def __init__(self, outcomes):
        self._outcomes = list(outcomes)
        self.calls = []  # model names, in call order
        self.uploads = []
        # The real client is client.models.generate_content / client.files.upload.
        self.models = self
        self.files = self

    def generate_content(self, *, model, contents, config=None):
        self.calls.append(model)
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        if isinstance(outcome, (bytes, bytearray)):
            return FakeAudioResponse(outcome)
        return FakeResponse(outcome)

    def upload(self, *, file):
        self.uploads.append(file)
        return "uploaded-file-handle"


@pytest.fixture
def settings():
    return Settings(models=("m1", "m2"), retries_per_model=2, base_delay_s=0.0, timeout_s=5)


@pytest.fixture
def sleeps():
    return []


def run(client, settings, sleeps, prompt="hi", image=None):
    # system_prompt="" skips building the SDK config object, keeping these pure.
    return ask_gemma(
        prompt,
        image,
        client=client,
        settings=settings,
        system_prompt="",
        sleep=sleeps.append,
    )


def test_first_try_success(settings, sleeps):
    client = FakeClient(["hello"])
    assert run(client, settings, sleeps) == Answer(text="hello", model="m1", attempts=1)
    assert sleeps == []


def test_transient_error_is_retried_on_same_model(settings, sleeps):
    client = FakeClient([FakeAPIError(503), "ok"])
    answer = run(client, settings, sleeps)
    assert (answer.model, answer.attempts) == ("m1", 2)
    assert client.calls == ["m1", "m1"]
    assert len(sleeps) == 1


def test_falls_back_to_next_model_after_retries_exhausted(settings, sleeps):
    client = FakeClient([FakeAPIError(500), FakeAPIError(503), "ok"])
    answer = run(client, settings, sleeps)
    assert answer.model == "m2"
    assert client.calls == ["m1", "m1", "m2"]


def test_raises_when_every_model_fails(settings, sleeps):
    client = FakeClient([FakeAPIError(503)] * 4)
    with pytest.raises(GemmaUnavailable) as info:
        run(client, settings, sleeps)
    assert "m1, m2" in str(info.value)
    assert info.value.last_error.code == 503


def test_non_transient_error_is_not_retried_or_fallen_back(settings, sleeps):
    client = FakeClient([FakeAPIError(400)])
    with pytest.raises(FakeAPIError):
        run(client, settings, sleeps)
    assert client.calls == ["m1"]


def test_unknown_model_skips_straight_to_next_without_waiting(settings, sleeps):
    client = FakeClient([FakeAPIError(404), "ok"])
    answer = run(client, settings, sleeps)
    assert answer.model == "m2"
    assert client.calls == ["m1", "m2"]
    assert sleeps == []


def test_empty_response_counts_as_a_failed_attempt(settings, sleeps):
    client = FakeClient(["", "ok"])
    answer = run(client, settings, sleeps)
    assert (answer.text, answer.attempts) == ("ok", 2)


def test_timeout_style_errors_are_retried(settings, sleeps):
    class ReadTimeout(Exception):  # name-matched, like httpx's
        pass

    client = FakeClient([ReadTimeout("slow"), "ok"])
    assert run(client, settings, sleeps).attempts == 2


def test_image_is_uploaded_once_across_retries_and_models(settings, sleeps, tmp_path):
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"not really a jpeg")
    client = FakeClient([FakeAPIError(503), FakeAPIError(503), "ok"])
    answer = run(client, settings, sleeps, image=image)
    assert answer.model == "m2"
    assert client.uploads == [str(image)]


def test_backoff_doubles_between_retries(sleeps):
    settings = Settings(models=("m1",), retries_per_model=3, base_delay_s=1.0)
    client = FakeClient([FakeAPIError(503), FakeAPIError(503), "ok"])
    run(client, settings, sleeps)
    # delay = base * 2**(attempt-1) plus up to base/2 of jitter
    assert 1.0 <= sleeps[0] < 1.5
    assert 2.0 <= sleeps[1] < 2.5


def test_settings_read_model_list_from_env(monkeypatch):
    monkeypatch.setenv("GEMMA_MODELS", "a, b ,c")
    assert Settings.from_env().models == ("a", "b", "c")


def test_settings_default_models_when_env_unset(monkeypatch):
    monkeypatch.delenv("GEMMA_MODELS", raising=False)
    assert Settings.from_env().models == ("gemma-4-26b-a4b-it", "gemma-4-31b-it")


def test_system_prompt_file_is_present_and_nonempty():
    assert load_system_prompt()

# ---------------------------------------------------------------------------
# Tests for the image-to-audio pipeline
# ---------------------------------------------------------------------------

AUDIO_SETTINGS = Settings(
    models=("vision-m1", "vision-m2"),
    retries_per_model=2,
    base_delay_s=0.0,
    timeout_s=5,
)
AUDIO_BYTES = b"RIFF\x00\x00\x00\x00WAVEfmt "  # minimal fake WAV header


def run_vision(client, settings, image, sleeps=None):
    return describe_image(
        image,
        client=client,
        settings=settings,
        sleep=(sleeps or []).append,
    )


def run_audio(client, settings, keyword, sleeps=None, audio_models=None):
    return generate_audio(
        keyword,
        client=client,
        settings=settings,
        audio_models=audio_models,
        sleep=(sleeps or []).append,
    )


def run_pipeline(client, settings, image, sleeps=None, audio_models=None):
    """Wrap image_to_audio so we can inject the fake client and audio models."""
    # image_to_audio doesn't accept audio_models directly; we reach in via
    # generate_audio which reads SOUND_AUDIO_MODELS or the kwarg.
    import app.gemma as gm

    sleeps = sleeps or []
    keyword_answer = gm.describe_image(
        image, client=client, settings=settings, sleep=sleeps.append
    )
    audio_bytes, mime, audio_model, audio_attempts = gm.generate_audio(
        keyword_answer.text,
        client=client,
        settings=settings,
        audio_models=audio_models,
        sleep=sleeps.append,
    )
    return gm.AudioResult(
        audio_bytes=audio_bytes,
        mime_type=mime,
        keyword=keyword_answer.text,
        vision_model=keyword_answer.model,
        audio_model=audio_model,
        total_attempts=keyword_answer.attempts + audio_attempts,
    )


# --- describe_image ---

def test_describe_image_success(tmp_path):
    image = tmp_path / "lion.jpg"
    image.write_bytes(b"fake-jpeg")
    client = FakeClient(["Deep lion roar"])
    answer = run_vision(client, AUDIO_SETTINGS, image)
    assert answer.text == "Deep lion roar"
    assert answer.model == "vision-m1"
    assert answer.attempts == 1
    assert client.uploads == [str(image)]


def test_describe_image_retries_transient(tmp_path):
    image = tmp_path / "lion.jpg"
    image.write_bytes(b"x")
    sleeps = []
    client = FakeClient([FakeAPIError(503), "Wind howling"])
    answer = run_vision(client, AUDIO_SETTINGS, image, sleeps)
    assert answer.text == "Wind howling"
    assert answer.attempts == 2
    assert len(sleeps) == 1


def test_describe_image_falls_back_to_second_model(tmp_path):
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"x")
    client = FakeClient([FakeAPIError(500), FakeAPIError(503), "Thunderstorm"])
    answer = run_vision(client, AUDIO_SETTINGS, image)
    assert answer.model == "vision-m2"
    assert client.calls == ["vision-m1", "vision-m1", "vision-m2"]


def test_describe_image_skips_404_immediately(tmp_path):
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"x")
    sleeps = []
    client = FakeClient([FakeAPIError(404), "Ocean waves"])
    answer = run_vision(client, AUDIO_SETTINGS, image, sleeps)
    assert answer.model == "vision-m2"
    assert sleeps == []  # no back-off on 404


def test_describe_image_raises_when_all_models_fail(tmp_path):
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"x")
    client = FakeClient([FakeAPIError(503)] * 4)
    with pytest.raises(GemmaUnavailable):
        run_vision(client, AUDIO_SETTINGS, image)


def test_describe_image_uploads_once_across_retries(tmp_path):
    image = tmp_path / "lion.jpg"
    image.write_bytes(b"x")
    client = FakeClient([FakeAPIError(503), FakeAPIError(503), "Roar"])
    run_vision(client, AUDIO_SETTINGS, image)
    assert client.uploads == [str(image)]  # uploaded exactly once


# --- generate_audio ---

def test_generate_audio_success():
    client = FakeClient([AUDIO_BYTES])
    data, mime, model, attempts = run_audio(
        client, AUDIO_SETTINGS, "Deep lion roar",
        audio_models=("flash-tts", "muse-glimmer"),
    )
    assert data == AUDIO_BYTES
    assert mime == "audio/wav"
    assert model == "flash-tts"
    assert attempts == 1


def test_generate_audio_falls_back_to_muse_glimmer():
    client = FakeClient([FakeAPIError(503), FakeAPIError(503), AUDIO_BYTES])
    data, mime, model, attempts = run_audio(
        client, AUDIO_SETTINGS, "Roar",
        audio_models=("flash-tts", "muse-glimmer"),
    )
    assert model == "muse-glimmer"
    assert client.calls == ["flash-tts", "flash-tts", "muse-glimmer"]


def test_generate_audio_skips_404_to_fallback():
    sleeps = []
    client = FakeClient([FakeAPIError(404), AUDIO_BYTES])
    _, _, model, _ = run_audio(
        client, AUDIO_SETTINGS, "Wind",
        audio_models=("flash-tts", "muse-glimmer"),
        sleeps=sleeps,
    )
    assert model == "muse-glimmer"
    assert sleeps == []


def test_generate_audio_raises_when_all_models_fail():
    client = FakeClient([FakeAPIError(503)] * 4)
    with pytest.raises(GemmaUnavailable) as info:
        run_audio(
            client, AUDIO_SETTINGS, "Rain",
            audio_models=("flash-tts", "muse-glimmer"),
        )
    assert "audio" in str(info.value).lower()


# --- full pipeline ---

def test_image_to_audio_pipeline(tmp_path):
    image = tmp_path / "lion.jpg"
    image.write_bytes(b"fake-jpeg")
    # Vision model succeeds on first try; audio model succeeds on first try.
    client = FakeClient(["Deep lion roar", AUDIO_BYTES])
    result = run_pipeline(
        client, AUDIO_SETTINGS, image,
        audio_models=("flash-tts", "muse-glimmer"),
    )
    assert isinstance(result, AudioResult)
    assert result.keyword == "Deep lion roar"
    assert result.vision_model == "vision-m1"
    assert result.audio_model == "flash-tts"
    assert result.audio_bytes == AUDIO_BYTES
    assert result.total_attempts == 2  # 1 vision + 1 audio


def test_pipeline_audio_fallback_counted_in_total_attempts(tmp_path):
    image = tmp_path / "lion.jpg"
    image.write_bytes(b"x")
    # Vision OK (1 attempt), audio needs 1 retry then falls back (3 attempts).
    client = FakeClient(["Roar", FakeAPIError(503), FakeAPIError(503), AUDIO_BYTES])
    result = run_pipeline(
        client, AUDIO_SETTINGS, image,
        audio_models=("flash-tts", "muse-glimmer"),
    )
    assert result.audio_model == "muse-glimmer"
    assert result.total_attempts == 1 + 3  # 1 vision + 3 audio

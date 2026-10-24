# 2. Retry transient errors, then fall back to the other Gemma model

Status: accepted

## Context

While building this, plain requests to both Gemma models returned
`500 INTERNAL`, one hung for 1m45s before failing, and a Gemini model returned
`503 UNAVAILABLE` ("high demand"). The request format and key were fine. This
is the failure a demo will actually hit.

## Decision

`ask_gemma()` in `app/gemma.py` does the following:

1. Retries 429/5xx errors and network timeouts up to `GEMMA_RETRIES_PER_MODEL`
   times (default 2) with exponential backoff plus jitter.
2. When a model is exhausted, moves to the next in `GEMMA_MODELS`
   (default: 26B A4B, then 31B) without waiting.
3. Skips a model immediately on 404, since the key can't see it.
4. Raises immediately on anything else (400, 401, 403): a bad key or a
   malformed request fails the same way on every model.
5. Sets an explicit client timeout (`GEMMA_TIMEOUT_S`, default 30) so a stuck
   request can't hang the UI.

If every model fails, it raises `GemmaUnavailable`, which the server turns into
a 503 with a plain-language message.

## Consequences

- Worst case, a request takes roughly
  `models x retries x timeout` plus backoff. With the defaults that is well
  over a minute. Lower `GEMMA_TIMEOUT_S` for a snappier demo.
- Both models likely share infrastructure, so a platform-wide outage defeats
  the fallback. It helps with a bad moment on one model, not a bad day.
- The retry policy is tested offline with a scripted fake client
  (`tests/test_gemma.py`), so changes to it are safe to make.

## Not done

- No circuit breaker or request queue. Not worth it until there is real traffic.
- No local-model fallback. It would keep the demo alive but would no longer be
  "Gemma through the Gemini API", which the challenge requires.

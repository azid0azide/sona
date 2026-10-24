"""Check the Gemma API end to end before a demo.

    python -m scripts.smoke                 # text only
    python -m scripts.smoke photo.jpg       # text + image

Prints which model answered and how many attempts it took, or exits non-zero
with the reason. Run it from the venue network, not just from home: the API
was flaky enough during development that this is worth thirty seconds.
"""

import sys

import app  # noqa: F401 - loads .env
from app.gemma import GemmaUnavailable, ask_gemma


def main() -> int:
    image = sys.argv[1] if len(sys.argv) > 1 else None
    prompt = "Describe this image in one sentence." if image else "Reply with the word: ready"
    try:
        answer = ask_gemma(prompt, image)
    except GemmaUnavailable as exc:
        print(f"FAILED: {exc}\nlast error: {exc.last_error!r}", file=sys.stderr)
        return 1
    print(f"model={answer.model} attempts={answer.attempts}\n{answer.text}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

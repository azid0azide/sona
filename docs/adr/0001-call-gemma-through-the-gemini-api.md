# 1. Call Gemma through the Gemini API instead of running the weights

Status: accepted

## Context

The prototype ran in Colab by loading Gemma 4 E2B with Transformers. That needs
a ~10 GB weights download and a GPU (a T4 works; a TPU runtime silently falls
back to a CPU crawl, and one image reply took over ten minutes). Production
would need the same, plus somewhere to host a GPU.

The Gemini API also hosts Gemma 4 (`gemma-4-26b-a4b-it` and `gemma-4-31b-it`).
The Google Gemma challenge at Hacktoberfest Hack Days additionally requires the
winning project to use a Gemma model through the Gemini API.

## Decision

The app calls Gemma through the Gemini API. No weights are downloaded or served
by this project, so the server needs no GPU and very little CPU.

Colab stays as the place to iterate on prompts, but it calls the same API
rather than loading a model.

## Consequences

- Hosting is a small Python web service on any ordinary host.
- We depend on API availability. On 2026-10-13 the API returned 500 on both
  Gemma models and 503 on a Gemini model, even for plain prompts. That is
  handled in ADR 0002, not ignored.
- We can't fine-tune or run offline with this setup. If either becomes a
  requirement, the model call is isolated in `app/gemma.py` so a local backend
  can replace it without touching the rest of the app.

## Alternatives considered

- **Ollama or LM Studio on the server**: documented and scriptable, but needs a
  machine capable of running the model, which is the cost we're avoiding.
- **Transformers on a cloud GPU**: most flexible, most expensive, and nothing
  in the project needs that flexibility yet.
- **Vertex AI**: a managed option for scale; worth revisiting if quotas become
  a problem, but more setup than a prototype warrants.

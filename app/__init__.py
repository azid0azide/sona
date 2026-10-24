from dotenv import load_dotenv

# Loaded here, once, so every entry point (uvicorn, scripts/smoke.py, pytest)
# sees the same environment without each one remembering to call it.
load_dotenv()

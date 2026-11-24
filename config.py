"""Runtime configuration."""
import os
from pathlib import Path


def _load_dotenv():
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = value


_load_dotenv()

API_KEY = os.environ.get("OPENAI_API_KEY", "")
MODEL = os.environ.get("OPENAI_MODEL") or os.environ.get("MODEL", "")
BASE_URL = os.environ.get("OPENAI_BASE_URL") or os.environ.get("BASE_URL", "")


def require_api_key() -> str:
    if not API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return API_KEY


MAX_STEPS = 10

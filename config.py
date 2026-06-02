from __future__ import annotations
import os
from pathlib import Path

def _load_dotenv() -> None:
    env_path = Path(__file__).resolve().parent / '.env'
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        (key, _, value) = line.partition('=')
        key = key.strip()
        value = value.strip().strip('\'"')
        if key and key not in os.environ:
            os.environ[key] = value
_load_dotenv()
API_KEY = os.environ.get('OPENAI_API_KEY', '')
if not API_KEY:
    raise RuntimeError('OPENAI_API_KEY is required.')
BASE_URL = os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com/v1')
AGENT_MODEL = os.environ.get('RISA_AGENT_MODEL', os.environ.get('OPENAI_MODEL', 'gpt-4.1-mini'))
GUARD_MODEL = os.environ.get('RISA_GUARD_MODEL', AGENT_MODEL)
MODEL = AGENT_MODEL
MAX_STEPS = int(os.environ.get('RISA_MAX_STEPS', '10'))

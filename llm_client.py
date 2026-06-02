from __future__ import annotations
import asyncio
import logging
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, AuthenticationError, InternalServerError, PermissionDeniedError, RateLimitError
from config import AGENT_MODEL, API_KEY, BASE_URL, GUARD_MODEL
logger = logging.getLogger(__name__)
_token_usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0, 'calls': 0}
_runtime_agent_model: str | None = None
_runtime_guard_model: str | None = None
_runtime_api_key: str | None = None
_runtime_base_url: str | None = None

def reset_token_usage() -> None:
    for key in _token_usage:
        _token_usage[key] = 0

def get_token_usage() -> dict:
    return dict(_token_usage)

def set_runtime_config(*, agent_model: str | None=None, guard_model: str | None=None, api_key: str | None=None, base_url: str | None=None) -> None:
    global _runtime_agent_model, _runtime_guard_model, _runtime_api_key, _runtime_base_url
    _runtime_agent_model = agent_model
    _runtime_guard_model = guard_model
    _runtime_api_key = api_key
    _runtime_base_url = base_url
    _ClientHolder.reset()

def get_agent_model() -> str:
    return _runtime_agent_model or AGENT_MODEL

def get_guard_model() -> str:
    return _runtime_guard_model or GUARD_MODEL

class _ClientHolder:
    _client: AsyncOpenAI | None = None

    @classmethod
    def get(cls) -> AsyncOpenAI:
        if cls._client is None:
            cls._client = AsyncOpenAI(api_key=_runtime_api_key or API_KEY, base_url=_runtime_base_url or BASE_URL, timeout=60)
        return cls._client

    @classmethod
    def reset(cls) -> None:
        cls._client = None

async def chat_completion(messages: list, tools: list | None=None, retries: int=5, role: str='agent'):
    model = get_guard_model() if role == 'guard' else get_agent_model()
    client = _ClientHolder.get()
    for attempt in range(retries):
        try:
            kwargs = {'model': model, 'messages': messages}
            if tools is not None:
                kwargs['tools'] = tools
            response = await client.chat.completions.create(**kwargs)
            usage = getattr(response, 'usage', None)
            if usage:
                _token_usage['prompt_tokens'] += getattr(usage, 'prompt_tokens', 0) or 0
                _token_usage['completion_tokens'] += getattr(usage, 'completion_tokens', 0) or 0
                _token_usage['total_tokens'] += getattr(usage, 'total_tokens', 0) or 0
                _token_usage['calls'] += 1
            return response
        except (AuthenticationError, PermissionDeniedError):
            raise
        except (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError) as exc:
            if attempt == retries - 1:
                raise
            wait_seconds = min(2 ** attempt, 16)
            logger.warning('OpenAI request failed (%s); retrying in %ss', type(exc).__name__, wait_seconds)
            await asyncio.sleep(wait_seconds)

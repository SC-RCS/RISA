"""Async LLM client wrapper."""
from __future__ import annotations

import asyncio
import json
import logging
import random
from openai import AsyncOpenAI, AuthenticationError, BadRequestError, PermissionDeniedError
try:
    from .config import API_KEY, BASE_URL, MODEL, require_api_key
except ImportError:
    from config import API_KEY, BASE_URL, MODEL, require_api_key

logger = logging.getLogger(__name__)

_token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0}


def reset_token_usage():
    _token_usage["prompt_tokens"] = 0
    _token_usage["completion_tokens"] = 0
    _token_usage["total_tokens"] = 0
    _token_usage["calls"] = 0


def get_token_usage() -> dict:
    return dict(_token_usage)


_agent_model_override: str | None = None
_runtime_model: str | None = None
_runtime_api_key: str | None = None
_runtime_base_url: str | None = None

_multi_keys: list[str] = []
_multi_clients: list[AsyncOpenAI] = []
_multi_semaphores: list[asyncio.Semaphore] = []
_multi_key_index = 0
_multi_key_lock = None


def set_agent_model(model_name: str | None):
    """Set the model used by the agent."""
    global _agent_model_override
    _agent_model_override = model_name
    LLMClient.reset()
    if model_name:
        logger.info(f"Agent model switched to: {model_name}  (Guard still uses: {_get_base_model()})")


def set_runtime_config(model: str | None = None, api_key: str | None = None, base_url: str | None = None):
    """Switch runtime model and API config (Agent and Guard both switch)"""
    global _runtime_model, _runtime_api_key, _runtime_base_url, _agent_model_override
    _runtime_model = model
    _runtime_api_key = api_key
    _runtime_base_url = base_url
    _agent_model_override = None
    LLMClient.reset()
    if model:
        logger.info(f"Runtime config switched: model={model}, base_url={base_url}")


def set_multi_keys(keys: list[str], base_url: str | None = None, concurrency_per_key: int = 1):
    """Configure multi-API-key rotation, each key with independent semaphore

    Args:
        keys: list of API keys
        base_url: shared base_url (defaults to config/runtime value)
        concurrency_per_key: max concurrency per key
    """
    global _multi_keys, _multi_clients, _multi_semaphores, _multi_key_index, _multi_key_lock
    url = base_url or _runtime_base_url or BASE_URL
    _multi_keys = list(keys)
    _multi_clients = [AsyncOpenAI(api_key=k, base_url=url) for k in keys]
    _multi_semaphores = [asyncio.Semaphore(concurrency_per_key) for _ in keys]
    _multi_key_index = 0
    _multi_key_lock = asyncio.Lock()
    logger.info(f"Multi-key rotation configured: {len(keys)} keys, concurrency_per_key={concurrency_per_key}, base_url={url}")


def is_multi_key_mode() -> bool:
    """Whether multi-key rotation mode is active"""
    return len(_multi_keys) > 0


async def _acquire_multi_key() -> tuple[int, AsyncOpenAI, asyncio.Semaphore]:
    """Round-robin acquire a key's client and semaphore"""
    global _multi_key_index
    async with _multi_key_lock:
        idx = _multi_key_index % len(_multi_keys)
        _multi_key_index += 1
    await _multi_semaphores[idx].acquire()
    return idx, _multi_clients[idx], _multi_semaphores[idx]


def _release_multi_key(idx: int):
    """Release the key's semaphore"""
    _multi_semaphores[idx].release()


def _get_base_model() -> str:
    return _runtime_model or MODEL


def get_agent_model() -> str:
    """Get the model name currently used by Agent"""
    return _agent_model_override or _runtime_model or MODEL


class LLMClient:
    """Async LLM client singleton"""
    _instance = None

    @classmethod
    def get(cls) -> AsyncOpenAI:
        if cls._instance is None:
            key = _runtime_api_key or require_api_key()
            url = _runtime_base_url or BASE_URL
            cls._instance = AsyncOpenAI(api_key=key, base_url=url)
        return cls._instance

    @classmethod
    def reset(cls):
        cls._instance = None


async def chat_completion(messages: list, tools: list | None = None, retries: int = 8, role: str = "agent"):
    """
    Call LLM with retry mechanism
    
    Args:
        role: "agent" uses agent model (may be a weak model), "guard" always uses strong model
    """
    model = get_agent_model() if role == "agent" else _get_base_model()

    if is_multi_key_mode():
        return await _chat_completion_multi_key(messages, tools, retries, model)

    for attempt in range(retries):
        client = LLMClient.get()
        try:
            kwargs = {"model": model, "messages": messages, "timeout": 60}
            if tools:
                kwargs["tools"] = tools
            response = await client.chat.completions.create(**kwargs)
            if not response.choices:
                raise ValueError("LLM returned empty choices list")
            if hasattr(response, 'usage') and response.usage:
                _token_usage["prompt_tokens"] += getattr(response.usage, 'prompt_tokens', 0) or 0
                _token_usage["completion_tokens"] += getattr(response.usage, 'completion_tokens', 0) or 0
                _token_usage["total_tokens"] += getattr(response.usage, 'total_tokens', 0) or 0
                _token_usage["calls"] += 1
            return response
        except (AuthenticationError, PermissionDeniedError) as e:
            raise
        except BadRequestError:
            raise
        except Exception as e:
            if attempt == retries - 1:
                raise
            backoff = min(2 ** attempt, 16) + random.uniform(0, 1)
            logger.warning(f"LLM call failed (attempt {attempt+1}): {e}, retrying in {backoff:.1f}s...")
            await asyncio.sleep(backoff)


async def _chat_completion_multi_key(messages: list, tools: list | None, retries: int, model: str):
    """chat_completion implementation for multi-key rotation mode"""
    for attempt in range(retries):
        idx, client, _ = await _acquire_multi_key()
        try:
            kwargs = {"model": model, "messages": messages, "timeout": 60}
            if tools:
                kwargs["tools"] = tools
            response = await client.chat.completions.create(**kwargs)
            if not response.choices:
                raise ValueError("LLM returned empty choices list")
            if hasattr(response, 'usage') and response.usage:
                _token_usage["prompt_tokens"] += getattr(response.usage, 'prompt_tokens', 0) or 0
                _token_usage["completion_tokens"] += getattr(response.usage, 'completion_tokens', 0) or 0
                _token_usage["total_tokens"] += getattr(response.usage, 'total_tokens', 0) or 0
                _token_usage["calls"] += 1
            return response
        except (AuthenticationError, PermissionDeniedError):
            raise
        except BadRequestError:
            raise
        except Exception as e:
            if attempt == retries - 1:
                raise
            backoff = min(2 ** attempt, 16) + random.uniform(0, 1)
            logger.warning(f"LLM call failed (key#{idx}, attempt {attempt+1}): {e}, retrying in {backoff:.1f}s...")
            await asyncio.sleep(backoff)
        finally:
            _release_multi_key(idx)

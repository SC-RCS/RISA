from __future__ import annotations
from collections.abc import Sequence
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.basic_elements import text_content_block_from_string
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.types import ChatMessage
from .executor import _extract_text_from_content

class FinalAnswerSafeguard(BasePipelineElement):
    name = 'final_answer_safeguard'

    def __init__(self, executor):
        self.executor = executor

    def query(self, query: str, runtime: FunctionsRuntime, env: Env=EmptyEnv(), messages: Sequence[ChatMessage]=[], extra_args: dict={}) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        if not messages:
            return (query, runtime, env, messages, extra_args)
        last = messages[-1]
        if last.get('role') != 'assistant' or last.get('tool_calls'):
            return (query, runtime, env, messages, extra_args)
        text = _extract_text_from_content(last.get('content', ''))
        if not text:
            return (query, runtime, env, messages, extra_args)
        (safe_text, reason) = self.executor.sanitize_final_answer(text)
        if not reason or safe_text == text:
            return (query, runtime, env, messages, extra_args)
        stats = getattr(self.executor, 'stats', None)
        if stats is not None:
            stats.final_answer_sanitize_count += 1
            stats.final_answer_sanitize_reasons.append(reason)
        new_last = dict(last)
        new_last['content'] = [text_content_block_from_string(safe_text)]
        extra = dict(extra_args or {})
        extra['risa_final_answer_safeguard'] = {'action': 'sanitized', 'reason': reason}
        return (query, runtime, env, [*messages[:-1], new_last], extra)

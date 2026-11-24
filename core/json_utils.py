"""Robust JSON parsing utilities."""
import json
import re
import logging

try:
    import json_repair
except ImportError:
    json_repair = None

logger = logging.getLogger(__name__)


def _strip_markdown_fence(text: str) -> str:
    """Remove markdown code block wrappers"""
    text = text.strip()
    if not text.startswith("```"):
        return text
    lines = text.split("\n")
    json_lines = []
    in_block = False
    for line in lines:
        if line.startswith("```") and not in_block:
            in_block = True
            continue
        elif line.startswith("```") and in_block:
            break
        elif in_block:
            json_lines.append(line)
    return "\n".join(json_lines) if json_lines else text


def _strip_think_blocks(text: str) -> str:
    """Remove provider-visible reasoning wrappers before JSON parsing."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def _iter_json_object_candidates(text: str):
    """Yield balanced JSON-object substrings from right to left."""
    spans = []
    stack = []
    in_string = False
    escape = False
    start = None
    for idx, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if not stack:
                start = idx
            stack.append(ch)
        elif ch == "}" and stack:
            stack.pop()
            if not stack and start is not None:
                spans.append((start, idx + 1))
                start = None
    for start, end in reversed(spans):
        yield text[start:end]


def robust_json_parse(raw_text: str, context: str = "") -> dict:
    """
    Robust JSON parsing — progressive strategy, guarantees returning a dict.

    Standard parsing.
    Repair. auto-fix (handles trailing commas, single quotes, comments, etc.)
    Extraction and repair.

    Args:
        raw_text: raw LLM output
        context:  call context label (for logging)

    Returns:
        parsed dict; returns empty dict if parsing completely fails
    """
    text = _strip_think_blocks(_strip_markdown_fence(raw_text))

    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    if json_repair:
        try:
            result = json_repair.loads(text)
            if isinstance(result, dict):
                return result
        except Exception:
            pass

    for candidate in _iter_json_object_candidates(text):
        try:
            result = json.loads(candidate)
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
        if json_repair:
            try:
                result = json_repair.loads(candidate)
                if isinstance(result, dict):
                    return result
            except Exception:
                pass

    logger.warning(f"[{context}] JSON parsing completely failed: {raw_text[:200]}")
    return {}

import json
import re
import logging
try:
    import json_repair
except ImportError:
    json_repair = None
logger = logging.getLogger(__name__)

def _strip_markdown_fence(text: str) -> str:
    text = text.strip()
    if not text.startswith('```'):
        return text
    lines = text.split('\n')
    json_lines = []
    in_block = False
    for line in lines:
        if line.startswith('```') and (not in_block):
            in_block = True
            continue
        elif line.startswith('```') and in_block:
            break
        elif in_block:
            json_lines.append(line)
    return '\n'.join(json_lines) if json_lines else text

def robust_json_parse(raw_text: str, context: str='') -> dict:
    text = _strip_markdown_fence(raw_text)
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
    match = re.search('\\{.*\\}', text, re.DOTALL)
    if match:
        candidate = match.group()
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
    logger.warning(f'[{context}] JSON解析完全失败: {raw_text[:200]}')
    return {}

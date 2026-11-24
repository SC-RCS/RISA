"""
Risk Signal Keyword Library & Detection Module

Standalone module, called by Guard audit and other stages.
Keywords are organized by behavioral categories.
"""

from __future__ import annotations

import re

RISK_INDICATORS_B1 = [
    "<|im_start|>", "<|im_end|>", "<|user|>", "<|endoftext|>",
    "<|start_header_id|>", "<|end_header_id|>", "<|EOT|>",
    "<|end_of_sentence|>", "</s>", "<role>",
    "[INST]", "[/INST]", "<end_of_turn>", "<start_of_turn>",
    "act as", "pretend to be", "you are now", "pretend",
    "ignore previous", "ignore all", "new instructions",
    "system prompt", "system override",
    "forget everything", "disable safety", "ignore safety",
    "developer mode", "bypass policy", "bypass guardrail", "bypass safety",
    "do not follow your rules", "always obey", "never refuse",
    "unconditionally comply", "without question",
    "until tokens are exhausted",
    "injection",
]

RISK_INDICATORS_B2 = [
    "off-topic", "unrelated task", "different objective",
    "separate goal", "side task", "hidden goal",
]

RISK_INDICATORS_B3 = [
    "sudo", "admin", "override", "bypass", "escalat",
    "root", "chmod 777", "0777", "grant root", "grant full access",
    "full system access", "execute any command", "without restriction",
    "rm -rf", "mkfs", "shred", "dd of=",
    "eval(", "exec(", "os.system", "subprocess",
    "curl | sh", "wget", "/dev/tcp", "netcat", "reverse shell",
    "fork bomb", "while true", "shutdown",
    "delete all",
    "exploit", "malicious", "attack", "unauthorized",
    "harmful", "illegal", "unsafe",
    "deceive", "fraud", "phishing", "social engineering",
]

RISK_INDICATORS_B4 = [
    "ignore warning",
]

RISK_INDICATORS_B5 = [
    "password", "api_key", "api-key", "secret", "credential",
    "private key", "private_key", "ssh_key", "id_rsa",
    "/etc/shadow", ".ssh/id_rsa", ".aws/credentials",
    ".env", ".kube/config", ".docker/config",
    ".bash_history", ".pgpass", "printenv",
    "reveal", "dump", "leak", "expose", "exfiltrate",
    "steal", "harvest",
    "hidden instructions", "conversation history",
]

RISK_INDICATORS_B6 = [
    "fabricat", "hallucin", "made up", "fake", "imagin",
    "misinformation", "conceal", "distort",
]

ALL_RISK_INDICATORS = (
    RISK_INDICATORS_B1 + RISK_INDICATORS_B2 + RISK_INDICATORS_B3 +
    RISK_INDICATORS_B4 + RISK_INDICATORS_B5 + RISK_INDICATORS_B6
)


def contains_risk_signal(text: str) -> bool:
    """Check whether the text contains risk signal keywords (case-insensitive)."""
    if not text:
        return False
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in ALL_RISK_INDICATORS)


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences by Chinese/English punctuation and newlines."""
    paragraphs = text.split("\n")
    sentences = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        parts = re.split(r"(?<=[。！？.!?])\s*", para)
        for p in parts:
            p = p.strip()
            if p:
                sentences.append(p)
    return sentences


def smart_truncate(
    text: str,
    max_chars: int = 2000,
    field_name: str = "",
    preserve_head: int = 200,
    preserve_tail: int = 100,
) -> str:
    """
    Risk-aware truncation: within the character budget, preferentially retain
    sentence segments that contain risk keywords.

    Logic:
      1. If text does not exceed the limit, return as-is.
      2. If it exceeds the limit:
         a) Keep the first preserve_head characters (head contains structural info).
         b) Scan the middle segment, split by sentences, and mark sentences containing risk keywords.
         c) Within the remaining budget, prioritize risk sentences, then fill with normal sentences.
         d) Keep the last preserve_tail characters (tail may contain conclusions / refusals).
    """
    if not text or len(text) <= max_chars:
        return text

    if max_chars < preserve_head + preserve_tail + 100:
        return text[:max_chars - 60] + f"\n... [Truncated, original text was {len(text)} characters]"

    head = text[:preserve_head]
    tail = text[-preserve_tail:] if preserve_tail else ""

    middle = text[preserve_head: -preserve_tail if preserve_tail else len(text)]
    sentences = _split_sentences(middle)

    risk_sents = []
    normal_sents = []
    for s in sentences:
        if contains_risk_signal(s):
            risk_sents.append(s)
        else:
            normal_sents.append(s)

    budget = max_chars - len(head) - len(tail) - 100
    assembled = []
    used = 0

    risk_included = 0
    for s in risk_sents:
        if used + len(s) + 1 > budget:
            break
        assembled.append(s)
        used += len(s) + 1
        risk_included += 1

    for s in normal_sents:
        if used + len(s) + 1 > budget:
            break
        assembled.append(s)
        used += len(s) + 1

    parts = [head]
    omitted = len(sentences) - len(assembled)
    if assembled:
        if risk_included > 0:
            parts.append(
                f"\n... [Prioritized {risk_included} risk-bearing segments out of {len(sentences)} sentences] ...\n"
            )
        elif omitted > 0:
            parts.append(f"\n... [Middle section truncated] ...\n")
        parts.append("\n".join(assembled))
    if omitted > 0:
        hint = f"({field_name})" if field_name else ""
        parts.append(
            f"\n... [Truncated{hint}, original text was {len(text)} characters, "
            f"omitted {omitted}/{len(sentences)} sentences, retained {risk_included} risk segments]"
        )
    if tail:
        parts.append("\n" + tail)

    return "".join(parts)

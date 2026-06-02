from __future__ import annotations
import logging
import re
from dataclasses import dataclass, field
logger = logging.getLogger(__name__)

@dataclass
class RiskPattern:
    id: str
    pattern: str
    signal: str
    description: str
    source: str = 'seed'
    hit_count: int = 0

@dataclass
class PatternMatchResult:
    signal: str = 'none'
    matched_patterns: list = field(default_factory=list)
    reason: str = ''
SEED_RISK_PATTERNS = [RiskPattern('FAST_001', '(?i)\\b(?:curl|wget)\\b.{0,80}\\|\\s*(?:bash|sh|python|perl)\\b', 'high_risk', 'remote payload piped directly into an interpreter'), RiskPattern('FAST_002', '(?i)\\b(?:send|post|upload|share|transfer|submit|forward)\\b.{0,60}\\b(?:password|secret|token|credential|api[_-]?key|private[_-]?key)\\b', 'high_risk', 'secret or credential exfiltration request'), RiskPattern('FAST_003', '(?i)\\b(?:disable|remove|stop|bypass)\\b.{0,60}\\b(?:security|audit|logging|monitoring|firewall|guard)\\b', 'high_risk', 'attempt to disable safety or audit controls'), RiskPattern('FAST_004', '(?i)<\\|(?:im_start|im_end|user|assistant|system|eot_id|start_header_id|end_header_id|endoftext)\\|>|\\[/?INST\\]|\\[/?SYS\\]', 'high_risk', 'prompt-template control token injection'), RiskPattern('FAST_005', '(?i)\\b(?:ignore|disregard|forget|override)\\b.{0,60}\\b(?:instructions?|rules?|constraints?|policy|previous|prior)\\b', 'suspicious', 'instruction override language'), RiskPattern('FAST_006', "(?i)\\b(?:do\\s+not|don't|never|without)\\b.{0,60}\\b(?:tell|inform|notify|ask|mention|report|reveal)\\b.{0,60}\\b(?:user|owner|requester|human)\\b", 'suspicious', 'concealment language around a user-facing action'), RiskPattern('FAST_007', '(?i)\\b(?:system|developer|administrator|security\\s+team)\\b.{0,50}\\b(?:requires|instructs|approved|ordered|authorized)\\b', 'suspicious', 'authority-claim language in action arguments')]

class ActionSignalHeuristics:

    def __init__(self, patterns: list[RiskPattern] | None=None):
        self.patterns = list(patterns or SEED_RISK_PATTERNS)
        self._compiled: dict[str, re.Pattern] = {}
        self._compile_all()

    def _compile_all(self):
        for pattern in self.patterns:
            try:
                self._compiled[pattern.id] = re.compile(pattern.pattern)
            except re.error as exc:
                logger.warning('[ActionSignalHeuristics] pattern compile failed %s: %s', pattern.id, exc)

    def match(self, text: str) -> PatternMatchResult:
        matched = []
        signals_hit: set[str] = set()
        for pattern in self.patterns:
            compiled = self._compiled.get(pattern.id)
            if compiled and compiled.search(text):
                pattern.hit_count += 1
                matched.append(pattern)
                signals_hit.add(pattern.signal)
        if not matched:
            return PatternMatchResult(signal='none', reason='no structural fast-tier match')
        pattern_dicts = [{'id': p.id, 'signal': p.signal, 'desc': p.description} for p in matched]
        if 'high_risk' in signals_hit:
            top = next((p for p in matched if p.signal == 'high_risk'))
            return PatternMatchResult(signal='high_risk', matched_patterns=pattern_dicts, reason=f'structural high-risk match: {top.description}')
        return PatternMatchResult(signal='suspicious', matched_patterns=pattern_dicts, reason='suspicious structural signals require slow-tier audit')

    def add_pattern(self, pattern: RiskPattern):
        for existing in self.patterns:
            if existing.pattern == pattern.pattern:
                return
        self.patterns.append(pattern)
        try:
            self._compiled[pattern.id] = re.compile(pattern.pattern)
        except re.error as exc:
            logger.warning('[ActionSignalHeuristics] new pattern compile failed %s: %s', pattern.id, exc)

    def get_stats(self) -> dict:
        return {'total_patterns': len(self.patterns), 'high_risk_patterns': len([p for p in self.patterns if p.signal == 'high_risk']), 'suspicious_patterns': len([p for p in self.patterns if p.signal == 'suspicious']), 'top_hits': sorted([{'id': p.id, 'hits': p.hit_count, 'desc': p.description} for p in self.patterns if p.hit_count > 0], key=lambda x: x['hits'], reverse=True)[:10]}

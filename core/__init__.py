"""RISA core modules for action authorization and observation safeguards."""

from .action_signal_heuristics import ActionSignalHeuristics
from .observation_quarantine import ObservationQuarantine
from .source_aware_action_router import SourceAwareActionRouter

__all__ = [
    "ActionSignalHeuristics",
    "FinalAnswerSafeguard",
    "ObservationQuarantine",
    "SourceAwareActionRouter",
    "SourceAwareGuardedExecutor",
]


def __getattr__(name):
    """Lazily import AgentDojo-dependent pipeline classes."""
    if name == "SourceAwareGuardedExecutor":
        from .executor import SourceAwareGuardedExecutor

        return SourceAwareGuardedExecutor
    if name == "FinalAnswerSafeguard":
        from .final_answer_safeguard import FinalAnswerSafeguard

        return FinalAnswerSafeguard
    raise AttributeError(name)

"""HITL strategies for involving auditors in the anomaly-detection loop.

Each strategy implements the BaseHITLStrategy interface, allowing the
experiment framework to compare them on the same data and reviewer budget.
"""

from .base import BaseHITLStrategy, ReviewBatch
from .no_hitl import NoHITLStrategy
from .threshold_adjustment import ThresholdAdjustmentStrategy
from .preference_model import PreferenceModelStrategy
from .active_learning import ActiveLearningStrategy
from .hybrid_scoring import HybridScoringStrategy
from .rule_mining import RuleMiningStrategy


STRATEGY_REGISTRY = {
    "no_hitl": NoHITLStrategy,
    "threshold_adjustment": ThresholdAdjustmentStrategy,
    "preference_model": PreferenceModelStrategy,
    "active_learning": ActiveLearningStrategy,
    "hybrid_scoring": HybridScoringStrategy,
    "rule_mining": RuleMiningStrategy,
}


def build_strategy(name: str, **kwargs) -> BaseHITLStrategy:
    if name not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown HITL strategy '{name}'. "
            f"Available: {sorted(STRATEGY_REGISTRY)}"
        )
    return STRATEGY_REGISTRY[name](**kwargs)


__all__ = [
    "BaseHITLStrategy",
    "ReviewBatch",
    "STRATEGY_REGISTRY",
    "build_strategy",
]

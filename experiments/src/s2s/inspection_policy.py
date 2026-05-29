"""Adaptive Inspection Policy — 3-tier threshold rule.

From Section III-C:
    σ >= τ_h  → skip inspection (level 0)
    τ_l <= σ < τ_h → quick L1 test (level 1)
    σ < τ_l  → full L2 test (level 2)
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class InspectionDecision:
    level: int          # 0=skip, 1=quick, 2=full
    cost: float         # USD
    time_min: float     # minutes


def adaptive_inspection(
    sigma: float,
    tau_h: float,
    tau_l: float,
    costs: dict,
) -> InspectionDecision:
    """Determine inspection depth from confidence score.

    Args:
        sigma: extractor confidence in [0, 1].
        tau_h: high threshold (skip if above).
        tau_l: low threshold (full inspect if below).
        costs: dict with keys 'skip', 'l1', 'l2', each having 'cost' and 'time_min'.
    """
    if sigma >= tau_h:
        c = costs.get("skip", {"cost": 0, "time_min": 0})
        return InspectionDecision(level=0, cost=c["cost"], time_min=c["time_min"])
    elif sigma >= tau_l:
        c = costs["l1"]
        return InspectionDecision(level=1, cost=c["cost"], time_min=c["time_min"])
    else:
        c = costs["l2"]
        return InspectionDecision(level=2, cost=c["cost"], time_min=c["time_min"])


def inspect_all(costs: dict) -> InspectionDecision:
    """Always full inspection (for opt-only / no-adaptive baselines)."""
    c = costs["l2"]
    return InspectionDecision(level=2, cost=c["cost"], time_min=c["time_min"])

"""Pipeline — one full run: config + seed -> RunMetrics.

This is the ONLY entry point for running experiments. Baselines are
selected by swapping extractor and policy, not by separate code paths.
"""

from __future__ import annotations  # allow PEP 604 (str | Path) on Python 3.9

import numpy as np
import yaml
from pathlib import Path

from .beta_model import s2s_update
from .inspection_policy import adaptive_inspection, inspect_all, InspectionDecision
from .decision_engine import greedy_allocate, random_allocate, fifo_allocate, threshold_allocate
from .metrics import compute_metrics, RunMetrics
from .extractors.base import AbstractExtractor, NullExtractor


def load_config(config_path: str | Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_pipeline(
    config: dict,
    extractor: AbstractExtractor,
    seed: int,
    use_adaptive_inspection: bool = True,
    capacity_fraction: float = 1.0,
    disable_greedy_ranking: bool = False,
    allocator: str = "greedy",
    asset_generator=None,
) -> RunMetrics:
    """Execute one full experiment run.

    Seed controls: batch composition, yield draws, demand realization.
    All RNG is derived from this single seed for reproducibility.

    Args:
        config: scenario config dict
        extractor: AbstractExtractor implementation
        seed: master RNG seed
        use_adaptive_inspection: if False, inspect all assets fully
        capacity_fraction: multiplier on baseline capacity (for sensitivity)
        asset_generator: callable(config, rng) -> list[dict], or None for default
    """
    # Master RNG — all randomness derives from here
    master_rng = np.random.default_rng(seed)

    # Split into independent streams
    gen_rng = np.random.default_rng(master_rng.integers(0, 2**31))
    extract_rng = np.random.default_rng(master_rng.integers(0, 2**31))
    decision_rng = np.random.default_rng(master_rng.integers(0, 2**31))

    # Generate assets
    if asset_generator is None:
        from ..data_generators.common import generate_assets
        assets = generate_assets(config, gen_rng)
    else:
        assets = asset_generator(config, gen_rng)

    # Extract phi, sigma for each asset (asset passed for structured-feature extractors)
    for a in assets:
        result = extractor.extract(a["text"], extract_rng, asset=a)
        a["phi"] = result.phi
        a["sigma"] = result.sigma

    # Inspection decisions
    insp_costs = config["inspection_costs"]
    tau_h = config["thresholds"]["tau_h"]
    tau_l = config["thresholds"]["tau_l"]

    for a in assets:
        if use_adaptive_inspection:
            a["inspection"] = adaptive_inspection(a["sigma"], tau_h, tau_l, insp_costs)
        else:
            a["inspection"] = inspect_all(insp_costs)

        # After inspection, phi is updated toward true condition
        # But only if extractor provided a meaningful estimate (phi < 1.0)
        # For opt-only (phi=1.0), inspection is a sunk cost — no info update
        if a["phi"] < 0.99:  # extractor provided meaningful signal
            if a["inspection"].level == 2:
                a["phi"] = 0.1 * a["phi"] + 0.9 * a["true_yield_factor"]
            elif a["inspection"].level == 1:
                a["phi"] = 0.5 * a["phi"] + 0.5 * a["true_yield_factor"]
        # phi=1.0 (no extractor): inspection paid but phi stays at 1.0
        # This models "inspect all without semantic context" = sunk cost

    # Capacity
    cap_key = "weekly_hours" if "weekly_hours" in config["capacity"] else "daily_hours"
    capacity_minutes = config["capacity"][cap_key] * 60 * capacity_fraction

    # Run decision engine (selected allocator)
    alloc_args = dict(
        assets=assets,
        capacity_minutes=capacity_minutes,
        prices=config["prices"],
        base_yields=config["base_yields"],
        processing_costs=config["processing_costs"],
        rng=decision_rng,
    )

    if allocator == "random":
        results = random_allocate(**alloc_args)
    elif allocator == "fifo":
        results = fifo_allocate(**alloc_args)
    elif allocator == "threshold":
        results = threshold_allocate(**alloc_args)
    else:  # "greedy" (default)
        results = greedy_allocate(
            **alloc_args,
            kappa_base=config.get("kappa_base", 10.0),
            gamma=config.get("gamma", 1.0),
            disable_ranking=disable_greedy_ranking,
        )

    # Compute metrics
    full_insp_cost = insp_costs["l2"]["cost"]
    return compute_metrics(results, full_insp_cost)

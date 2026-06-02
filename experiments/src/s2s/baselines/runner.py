"""Baseline allocators — each is a config preset for pipeline.run_pipeline().

All baselines run through the SAME pipeline code path. The only differences
are: which extractor is used, whether adaptive inspection is enabled, and
how phi/sigma are set.

This prevents code drift between baselines and the main system.
"""

import copy

import numpy as np
from ..extractors.base import AbstractExtractor, ExtractionResult, NullExtractor
from ..pipeline import run_pipeline
from ..metrics import RunMetrics

_STRUCTURED_TRAIN_SEED = 99999
_STRUCTURED_TRAIN_N = 2000


class RandomExtractor(AbstractExtractor):
    """Baseline 1: Random phi, no meaningful signal."""
    def extract(self, text: str, rng: np.random.Generator,
                asset: dict = None) -> ExtractionResult:
        return ExtractionResult(phi=rng.uniform(0.3, 1.0), sigma=rng.uniform(0.0, 1.0))


def train_structured_mapping(config: dict) -> dict:
    """Learn age_bracket -> mean(true_yield) on a held-out training population.

    This is the honest "XGBoost on structured features" mapping. It is trained
    on an independent population (fixed seed, NOT the evaluation seeds) and uses
    only the structured age_bracket feature, ignoring text.
    """
    from ...data_generators.common import generate_assets
    train_cfg = copy.deepcopy(config)
    train_cfg["n_assets"] = _STRUCTURED_TRAIN_N
    rng = np.random.default_rng(_STRUCTURED_TRAIN_SEED)
    assets = generate_assets(train_cfg, rng)

    by_bracket: dict = {}
    for a in assets:
        by_bracket.setdefault(a["age_bracket"], []).append(a["true_yield_factor"])
    return {b: float(np.mean(ys)) for b, ys in by_bracket.items()}


class StructuredOnlyExtractor(AbstractExtractor):
    """Baseline 3 (XGBoost proxy): phi from structured age feature, no text.

    Simulates XGBoost trained on structured features. The age->phi mapping is
    learned by train_structured_mapping() on a held-out 2000-asset population
    and applied to each test asset via its age_bracket. Text is ignored.
    """
    def __init__(self, age_to_phi: dict = None):
        # Trained mapping age_bracket -> phi. Falls back to a neutral 0.75 only
        # if an unseen bracket appears (should not happen after training).
        self._mapping = age_to_phi or {}

    def extract(self, text: str, rng: np.random.Generator,
                asset: dict = None) -> ExtractionResult:
        age_bracket = asset.get("age_bracket", 0) if asset else 0
        phi = self._mapping.get(age_bracket, 0.75)
        return ExtractionResult(phi=float(np.clip(phi, 0.01, 1.0)), sigma=0.99)


class LLMOnlyExtractor(AbstractExtractor):
    """Baseline 5: LLM-only (CoT). Uses phi from keyword extractor but
    ignores optimizer — disposition is threshold-based on phi alone."""
    def __init__(self, inner: AbstractExtractor):
        self._inner = inner

    def extract(self, text: str, rng: np.random.Generator,
                asset: dict = None) -> ExtractionResult:
        return self._inner.extract(text, rng, asset)


def run_baseline(
    baseline: str,
    config: dict,
    keyword_extractor: AbstractExtractor,
    seed: int,
    capacity_fraction: float = 1.0,
) -> RunMetrics:
    """Run a named baseline through the standard pipeline.

    Args:
        baseline: one of "random", "rule_based", "xgboost", "opt_only", "llm_only", "ours"
        config: scenario config
        keyword_extractor: the scenario's KeywordExtractor instance
        seed: master RNG seed
        capacity_fraction: for sensitivity analysis
    """
    if baseline == "random":
        # Random disposition: random allocator, no semantic info, no inspection
        return run_pipeline(config, RandomExtractor(), seed,
                          use_adaptive_inspection=False,
                          capacity_fraction=capacity_fraction,
                          allocator="random")

    elif baseline == "rule_based":
        # FIFO: process all at L1 in arrival order, inspect all
        return run_pipeline(config, NullExtractor(), seed,
                          use_adaptive_inspection=False,
                          capacity_fraction=capacity_fraction,
                          allocator="fifo")

    elif baseline == "xgboost":
        # Structured features only (age), greedy optimizer, inspect all.
        # Mapping trained on a held-out population, applied via each asset's age.
        mapping = train_structured_mapping(config)
        return run_pipeline(config, StructuredOnlyExtractor(mapping), seed,
                          use_adaptive_inspection=False,
                          capacity_fraction=capacity_fraction,
                          allocator="greedy")

    elif baseline == "opt_only":
        # phi=1.0, inspect all, greedy optimizer
        return run_pipeline(config, NullExtractor(), seed,
                          use_adaptive_inspection=False,
                          capacity_fraction=capacity_fraction,
                          allocator="greedy")

    elif baseline == "semantic_only":
        # Baseline 5: threshold routing from phi, no optimizer
        return run_pipeline(config, keyword_extractor, seed,
                          use_adaptive_inspection=True,
                          capacity_fraction=capacity_fraction,
                          allocator="threshold")

    elif baseline == "ours":
        return run_pipeline(config, keyword_extractor, seed,
                          use_adaptive_inspection=True, capacity_fraction=capacity_fraction)

    else:
        raise ValueError(f"Unknown baseline: {baseline}")

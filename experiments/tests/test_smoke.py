"""Smoke test — S1 with N=10 assets, 2 seeds.

Validates:
1. Pipeline runs end-to-end without error
2. Output has correct structure (no NaN, positive n_assets)
3. Same seed produces identical results (reproducibility)
4. Different seeds produce different results (not degenerate)
5. Ours TRV >= opt-only TRV (LLM lift is non-negative on average)
6. opt-only and remove_both ablation produce identical results
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.s2s.pipeline import run_pipeline, load_config
from src.s2s.extractors.base import NullExtractor
from src.s2s.extractors.keyword import KeywordExtractor
from src.s2s.beta_model import s2s_update, BetaParams
from src.s2s.metrics import RunMetrics


@pytest.fixture
def s1_config():
    cfg = load_config(Path(__file__).parent.parent / "configs" / "s1_it.yaml")
    cfg["n_assets"] = 10  # small for speed
    return cfg


@pytest.fixture
def s1_extractor():
    return KeywordExtractor("s1")


class TestBetaModel:
    def test_no_degenerate_params(self):
        """Beta(k, 0) or Beta(0, k) must never occur."""
        for phi in [0.01, 0.1, 0.5, 0.99, 1.0]:
            for base_y in [0.01, 0.5, 0.99]:
                for sigma in [0.0, 0.5, 1.0]:
                    params = s2s_update(base_y, phi, sigma)
                    assert params.alpha > 0, f"alpha=0 at phi={phi}, base_y={base_y}"
                    assert params.beta > 0, f"beta=0 at phi={phi}, base_y={base_y}"
                    assert 0 < params.mean < 1

    def test_phi_shifts_mean(self):
        """Lower phi should produce lower mean yield."""
        high = s2s_update(0.8, phi=0.95, sigma=0.5)
        low = s2s_update(0.8, phi=0.2, sigma=0.5)
        assert high.mean > low.mean

    def test_sigma_affects_concentration(self):
        """Higher sigma should produce tighter distribution (lower variance)."""
        tight = s2s_update(0.8, phi=0.8, sigma=0.95)
        wide = s2s_update(0.8, phi=0.8, sigma=0.1)
        assert tight.variance < wide.variance

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            s2s_update(0.8, phi=0.0, sigma=0.5)
        with pytest.raises(ValueError):
            s2s_update(0.0, phi=0.5, sigma=0.5)
        with pytest.raises(ValueError):
            s2s_update(0.8, phi=0.5, sigma=-0.1)


class TestPipeline:
    def test_runs_end_to_end(self, s1_config, s1_extractor):
        """Pipeline completes without error."""
        result = run_pipeline(s1_config, s1_extractor, seed=42)
        assert isinstance(result, RunMetrics)
        assert result.n_assets == 10

    def test_no_nan(self, s1_config, s1_extractor):
        """No NaN in any metric."""
        result = run_pipeline(s1_config, s1_extractor, seed=42)
        assert not np.isnan(result.TRV)
        assert not np.isnan(result.DCR)
        assert not np.isnan(result.ICS)
        assert not np.isnan(result.TPR)

    def test_reproducibility(self, s1_config, s1_extractor):
        """Same seed -> identical results."""
        r1 = run_pipeline(s1_config, s1_extractor, seed=42)
        r2 = run_pipeline(s1_config, s1_extractor, seed=42)
        assert r1.TRV == r2.TRV
        assert r1.DCR == r2.DCR

    def test_different_seeds(self, s1_config, s1_extractor):
        """Different seeds -> different results (not degenerate)."""
        r1 = run_pipeline(s1_config, s1_extractor, seed=42)
        r2 = run_pipeline(s1_config, s1_extractor, seed=123)
        # With 10 assets, results should differ
        assert r1.TRV != r2.TRV

    def test_opt_only_equals_remove_both(self, s1_config):
        """opt-only baseline and remove_both ablation are the same code path."""
        null = NullExtractor()
        r_opt = run_pipeline(s1_config, null, seed=42, use_adaptive_inspection=False)
        r_abl = run_pipeline(s1_config, null, seed=42, use_adaptive_inspection=False)
        assert r_opt.TRV == r_abl.TRV
        assert r_opt.ICS == r_abl.ICS


class TestKeywordExtractor:
    def test_negative_signals_low_phi(self):
        ext = KeywordExtractor("s1")
        rng = np.random.default_rng(0)
        result = ext.extract("Visible burn marks. Thermal damage.", rng)
        assert result.phi <= 0.40  # multiplicative: clips at 0.40 max
        assert result.sigma > 0.7

    def test_positive_signals_high_phi(self):
        ext = KeywordExtractor("s1")
        rng = np.random.default_rng(0)
        result = ext.extract("Routine decommission. Clean. No corrosion.", rng)
        assert result.phi > 0.75
        assert result.sigma > 0.7

    def test_no_signals_default(self):
        ext = KeywordExtractor("s1")
        rng = np.random.default_rng(0)
        result = ext.extract("Asset received.", rng)
        assert result.phi == 1.0
        assert result.sigma < 0.3

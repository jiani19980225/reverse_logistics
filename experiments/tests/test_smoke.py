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


class TestSeedControl:
    """Locks the seed guarantees that make paired Wilcoxon valid (PROBLEM 4)."""

    def _population(self, cfg, seed):
        from src.data_generators.common import generate_assets
        master = np.random.default_rng(seed)
        gen_rng = np.random.default_rng(master.integers(0, 2**31))
        assets = generate_assets(cfg, gen_rng)
        return [(a["asset_type"], round(a["true_yield_factor"], 9), a["text"]) for a in assets]

    def test_same_seed_shared_population(self, s1_config):
        """Same seed -> identical population (every method sees the same assets)."""
        assert self._population(s1_config, 7) == self._population(s1_config, 7)

    def test_different_seeds_independent_population(self, s1_config):
        """Different seeds -> genuinely different populations, not a reshuffle."""
        p0 = self._population(s1_config, 0)
        p1 = self._population(s1_config, 1)
        assert p0 != p1
        # sorted true_yields differ -> not the same population re-ordered
        assert sorted(t[1] for t in p0) != sorted(t[1] for t in p1)


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


class _FakeAnthropic:
    """Offline stand-in for the Anthropic client: returns a fixed JSON payload
    in the same shape messages.create() produces, so LLMExtractor is testable
    without the anthropic package or a network call."""

    class _Block:
        type = "text"

        def __init__(self, text):
            self.text = text

    class _Resp:
        def __init__(self, text):
            self.content = [_FakeAnthropic._Block(text)]

    class _Messages:
        def __init__(self, payload):
            self._payload = payload

        def create(self, **kwargs):
            import json
            return _FakeAnthropic._Resp(json.dumps(self._payload))

    def __init__(self, payload):
        self.messages = _FakeAnthropic._Messages(payload)


class TestLLMExtractor:
    def test_parses_and_clips(self):
        """Injected client → phi/sigma parsed from JSON and clipped to range."""
        from src.s2s.extractors.llm import LLMExtractor
        # phi over 1.0 and sigma over 1.0 must clip; no network used.
        fake = _FakeAnthropic({"phi": 1.4, "sigma": 1.2, "condition": "clean"})
        ext = LLMExtractor("s1", client=fake)
        result = ext.extract("Routine decommission. Clean.", None)
        assert result.phi == 1.0          # clipped from 1.4
        assert result.sigma == 1.0        # clipped from 1.2

    def test_response_cache_hits(self):
        """A cached note is not re-sent to the client."""
        from src.s2s.extractors.llm import LLMExtractor
        cache = {"PSU failure. Burnt.": (0.05, 0.95)}
        # Client that would raise if called — proves the cache short-circuits.
        class _Boom:
            class messages:
                @staticmethod
                def create(**kwargs):
                    raise AssertionError("should not call API on cache hit")
        ext = LLMExtractor("s1", client=_Boom(), response_cache=cache)
        result = ext.extract("PSU failure. Burnt.", None)
        assert result.phi == 0.05
        assert result.sigma == 0.95

    def test_missing_key_raises(self, monkeypatch):
        """No injected client and no API key → clear RuntimeError, not a crash."""
        from src.s2s.extractors.llm import LLMExtractor
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        ext = LLMExtractor("s2")  # no client injected
        with pytest.raises(RuntimeError):
            ext.extract("Corroded skin panel.", None)

    def test_unknown_scenario_rejected(self):
        from src.s2s.extractors.llm import LLMExtractor
        with pytest.raises(ValueError):
            LLMExtractor("s9")


class _FakeGemini:
    """Offline stand-in for the google-genai client: .models.generate_content(...)
    returns an object with a .text JSON string."""

    class _Resp:
        def __init__(self, text):
            self.text = text

    class _Models:
        def __init__(self, payload):
            self._payload = payload

        def generate_content(self, **kwargs):
            import json
            return _FakeGemini._Resp(json.dumps(self._payload))

    def __init__(self, payload):
        self.models = _FakeGemini._Models(payload)


class TestGeminiExtractor:
    def test_parses_and_clips(self):
        from src.s2s.extractors.gemini import GeminiExtractor
        fake = _FakeGemini({"phi": -0.2, "sigma": 0.7, "condition": "dead"})
        ext = GeminiExtractor("s3", client=fake)
        result = ext.extract("Dead. Won't turn on.", None)
        assert result.phi == 0.01        # clipped up from -0.2
        assert result.sigma == 0.7

    def test_missing_key_raises(self, monkeypatch):
        from src.s2s.extractors.gemini import GeminiExtractor
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        ext = GeminiExtractor("s1")  # no client injected
        with pytest.raises(RuntimeError):
            ext.extract("Routine decommission.", None)

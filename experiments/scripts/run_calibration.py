"""Calibration analysis — honest Pearson r between extractor output and ground truth.

Reproduces the numbers in the paper's "Calibration Analysis" section
(S1/S2/S3 keyword-extractor r). The correlation is computed on DECOUPLED data:
the keyword extractor sees only the (noisy) text and emits phi; ground truth is
true_yield_factor, which for S1 is generated independently of the note text
(see s1_generator.py, PROBLEM 1 fix). No number is targeted or tuned — we report
whatever the deterministic extractor produces.

Two correlations are reported per scenario:
  - r(phi, true_yield)   : does the context factor track the realized yield?
  - r(yhat, true_yield)  : phi*base_yield (the actual quantity used as the prior mean)

Pooled across the given seeds for a stable estimate; per-seed mean/std also shown.

Usage:
    python scripts/run_calibration.py --seeds 0-29
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.s2s.pipeline import load_config
from src.s2s.extractors.keyword import KeywordExtractor
from src.s2s.extractors.strong import StrongExtractor
from src.s2s.extractors.llm import LLMExtractor
from src.data_generators.common import generate_assets

SCENARIOS = [
    ("configs/s1_it.yaml", "s1", "S1: IT Infrastructure"),
    ("configs/s2_aviation.yaml", "s2", "S2: Aviation MRO"),
    ("configs/s3_consumer.yaml", "s3", "S3: Consumer Electronics"),
]


def _parse_seeds(spec: str) -> list[int]:
    if "-" in spec and "," not in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in spec.split(",")]


def calibrate_one(config: dict, scenario_key: str, seeds: list[int],
                  extractor=None) -> dict:
    """Generate assets, run an extractor, correlate phi vs ground truth.

    Mirrors how the pipeline derives its RNG streams so the text/yields scored
    here are the same ones the simulation uses. `extractor` defaults to the
    scenario KeywordExtractor.
    """
    ext = extractor if extractor is not None else KeywordExtractor(scenario_key)
    base_yields = config["base_yields"]

    phi_all, yhat_all, ytrue_all = [], [], []
    per_seed_r = []

    for seed in seeds:
        master_rng = np.random.default_rng(seed)
        gen_rng = np.random.default_rng(master_rng.integers(0, 2**31))
        extract_rng = np.random.default_rng(master_rng.integers(0, 2**31))
        # decision_rng is consumed in the pipeline; we mirror the split so the
        # generation/extraction streams match the simulation exactly.
        _ = np.random.default_rng(master_rng.integers(0, 2**31))

        assets = generate_assets(config, gen_rng)

        phi_s, yhat_s, ytrue_s = [], [], []
        for a in assets:
            res = ext.extract(a["text"], extract_rng)
            # mean base yield over the asset's components (the quantity scaled by phi)
            comp_by = [base_yields.get(c, 0.75) for c in a["components"]]
            mean_by = float(np.mean(comp_by)) if comp_by else 0.75
            phi_s.append(res.phi)
            yhat_s.append(res.phi * mean_by)
            ytrue_s.append(a["true_yield_factor"])

        phi_all += phi_s
        yhat_all += yhat_s
        ytrue_all += ytrue_s
        # per-seed r(phi, true) — guard against zero variance (all phi==1.0)
        if np.std(phi_s) > 1e-9 and np.std(ytrue_s) > 1e-9:
            per_seed_r.append(stats.pearsonr(phi_s, ytrue_s)[0])

    phi_all = np.array(phi_all)
    yhat_all = np.array(yhat_all)
    ytrue_all = np.array(ytrue_all)

    out = {
        "n": len(phi_all),
        "frac_fallback": float(np.mean(phi_all >= 0.999)),  # phi==1.0 (no signal)
    }
    if np.std(phi_all) > 1e-9:
        out["r_phi"] = float(stats.pearsonr(phi_all, ytrue_all)[0])
        out["r_yhat"] = float(stats.pearsonr(yhat_all, ytrue_all)[0])
        out["r_phi_per_seed_mean"] = float(np.mean(per_seed_r)) if per_seed_r else float("nan")
        out["r_phi_per_seed_std"] = float(np.std(per_seed_r)) if per_seed_r else float("nan")
    else:
        out["r_phi"] = float("nan")  # extractor emitted constant phi -> undefined
        out["r_yhat"] = float("nan")
        out["r_phi_per_seed_mean"] = float("nan")
        out["r_phi_per_seed_std"] = float("nan")
    return out


def _make_llm_extractor(provider: str, scenario_key: str, model: str, cache: dict):
    """Build the LLM extractor for the chosen provider, sharing a response cache."""
    if provider == "gemini":
        from src.s2s.extractors.gemini import GeminiExtractor
        return GeminiExtractor(scenario_key, model=model, response_cache=cache)
    if provider == "deepseek":
        from src.s2s.extractors.deepseek import DeepSeekExtractor
        return DeepSeekExtractor(scenario_key, model=model, response_cache=cache)
    if provider == "anthropic":
        return LLMExtractor(scenario_key, model=model, response_cache=cache)
    raise ValueError(f"Unknown LLM provider: {provider}")


def calibrate_llm(config: dict, scenario_key: str, seeds: list, sample: int,
                  cache_path: Path, model: str, provider: str) -> dict:
    """LLM calibration on a bounded, deterministic subsample of notes.

    Collects (note, true_yield) pairs across the seeds, subsamples `sample`
    records with a fixed RNG (so the same records are scored every run), runs the
    chosen LLM extractor, and correlates phi against true_yield. Responses are
    cached to `cache_path` so re-runs are free and reproducible. Requires the
    provider's SDK + API key (raises RuntimeError otherwise).
    """
    records = []
    for seed in seeds:
        master = np.random.default_rng(seed)
        gen_rng = np.random.default_rng(master.integers(0, 2**31))
        assets = generate_assets(config, gen_rng)
        for a in assets:
            records.append((a["text"], a["true_yield_factor"]))

    pick = np.random.default_rng(12345).permutation(len(records))[:sample]
    sampled = [records[i] for i in pick]

    cache = {}
    if cache_path and cache_path.exists():
        cache = {k: tuple(v) for k, v in json.loads(cache_path.read_text()).items()}

    ext = _make_llm_extractor(provider, scenario_key, model, cache)
    phis, ys = [], []
    for text, y in sampled:
        res = ext.extract(text, None)
        phis.append(res.phi)
        ys.append(y)

    if cache_path:
        cache_path.write_text(json.dumps({k: list(v) for k, v in cache.items()}))

    r = stats.pearsonr(phis, ys)[0] if np.std(phis) > 1e-9 else float("nan")
    return {"n": len(phis), "r_phi": float(r)}


_DEFAULT_LLM_MODEL = {
    "anthropic": "claude-opus-4-8",
    "gemini": "gemini-2.5-flash",
    "deepseek": "deepseek-chat",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-29")
    ap.add_argument("--llm", action="store_true",
                    help="Also run the optional LLM extractor (needs the provider "
                         "SDK + API key). Bounded subsample, cached.")
    ap.add_argument("--llm-provider", default="anthropic",
                    choices=["anthropic", "gemini", "deepseek"],
                    help="LLM provider: anthropic (ANTHROPIC_API_KEY) or "
                         "gemini (GEMINI_API_KEY). Default anthropic.")
    ap.add_argument("--llm-sample", type=int, default=60,
                    help="Records per scenario for the LLM study (default 60).")
    ap.add_argument("--llm-model", default=None,
                    help="Model ID (default: claude-opus-4-8 / gemini-2.5-flash).")
    args = ap.parse_args()
    seeds = _parse_seeds(args.seeds)
    base_dir = Path(__file__).parent.parent

    print("=" * 72)
    print(f"CALIBRATION ANALYSIS — extractor phi vs ground truth")
    print(f"(decoupled generation; pooled over {len(seeds)} seeds)")
    print("=" * 72)
    print(f"{'Scenario':<28} {'N':>6} {'r(phi,y)':>10} {'r(yhat,y)':>11} {'phi=1.0 frac':>13}")
    print("-" * 72)

    for cfg_path, key, label in SCENARIOS:
        cfg = load_config(base_dir / cfg_path)
        res = calibrate_one(cfg, key, seeds)
        rphi = f"{res['r_phi']:.3f}" if not np.isnan(res["r_phi"]) else "undefined"
        ryhat = f"{res['r_yhat']:.3f}" if not np.isnan(res["r_yhat"]) else "undefined"
        print(f"{label:<28} {res['n']:>6} {rphi:>10} {ryhat:>11} {res['frac_fallback']:>12.1%}")

    print("-" * 72)
    print("r(phi,y)  = Pearson r between extractor context factor phi and true_yield")
    print("r(yhat,y) = Pearson r between prior mean (phi*base_yield) and true_yield")
    print("'phi=1.0 frac' = share of assets where the extractor found no signal and")
    print("                 fell back to the uninformative prior (phi=1.0).")

    # ---- Extractor-quality ladder on the SYNTHETIC data (Table V) ----
    # Deterministic references on the same decoupled notes:
    #   Keyword       = conservative lower bound (brittle vocabulary)
    #   Phrase-matcher= complete vocabulary, discrete phi per condition
    # An LLM (run with --llm) typically exceeds the phrase matcher, so the
    # phrase matcher is a reference point, NOT an absolute ceiling.
    print()
    print("=" * 72)
    print("EXTRACTOR-QUALITY LADDER (synthetic benchmark) — Table V")
    print("=" * 72)
    print(f"{'Scenario':<28} {'N':>6} {'Keyword r':>12} {'Phrase-matcher r':>18}")
    print("-" * 72)
    for cfg_path, key, label in SCENARIOS:
        cfg = load_config(base_dir / cfg_path)
        kw = calibrate_one(cfg, key, seeds, KeywordExtractor(key))
        st = calibrate_one(cfg, key, seeds, StrongExtractor(key))
        kw_r = f"{kw['r_phi']:.3f}" if not np.isnan(kw["r_phi"]) else "undefined"
        st_r = f"{st['r_phi']:.3f}" if not np.isnan(st["r_phi"]) else "undefined"
        print(f"{label:<28} {kw['n']:>6} {kw_r:>12} {st_r:>18}")
    print("-" * 72)
    print("Keyword r        = conservative lower bound (vocabulary tuned to stylized notes).")
    print("Phrase-matcher r = complete vocabulary, discrete phi per condition.")
    print("Both are deterministic and bounded below 1.0 by note<->condition decoupling noise.")
    print("Run with --llm to add a (typically higher) LLM column.")

    # ---- Optional LLM extractor study (opt-in; needs provider SDK + API key) ----
    if args.llm:
        model = args.llm_model or _DEFAULT_LLM_MODEL[args.llm_provider]
        print()
        print("=" * 72)
        print(f"LLM EXTRACTOR ({args.llm_provider}: {model}) — bounded subsample, cached")
        print("=" * 72)
        cache_dir = base_dir / "outputs" / "llm_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            print(f"{'Scenario':<28} {'N':>6} {'LLM r(phi,y)':>14}")
            print("-" * 72)
            for cfg_path, key, label in SCENARIOS:
                cfg = load_config(base_dir / cfg_path)
                cache_path = cache_dir / f"{key}_{args.llm_provider}_{model}.json"
                res = calibrate_llm(cfg, key, seeds, args.llm_sample,
                                    cache_path, model, args.llm_provider)
                rphi = f"{res['r_phi']:.3f}" if not np.isnan(res["r_phi"]) else "undefined"
                print(f"{label:<28} {res['n']:>6} {rphi:>14}")
            print("-" * 72)
            print(f"LLM r on the synthetic benchmark ({args.llm_sample} notes/scenario).")
            print("Responses cached under outputs/llm_cache/ for reproducible re-runs.")
        except RuntimeError as e:
            print(f"[skipped] {e}")


if __name__ == "__main__":
    main()

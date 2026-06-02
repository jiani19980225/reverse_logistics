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
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.s2s.pipeline import load_config
from src.s2s.extractors.keyword import KeywordExtractor
from src.s2s.extractors.strong import StrongExtractor
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-29")
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

    # ---- Table V replacement: extractor-quality bracket on the SYNTHETIC data ----
    # Honest, reproducible stand-in for the unsubstantiated real-text/LLM r=0.88.
    # Keyword = lower bound (brittle vocabulary); Strong = upper bound (oracle text
    # reader). Both run on the same synthetic, decoupled benchmark. NOT an LLM and
    # NOT real public text.
    print()
    print("=" * 72)
    print("EXTRACTOR-QUALITY BRACKET (synthetic benchmark) — replaces old Table V")
    print("=" * 72)
    print(f"{'Scenario':<28} {'N':>6} {'Keyword r':>12} {'Strong r':>11}")
    print("-" * 72)
    for cfg_path, key, label in SCENARIOS:
        cfg = load_config(base_dir / cfg_path)
        kw = calibrate_one(cfg, key, seeds, KeywordExtractor(key))
        st = calibrate_one(cfg, key, seeds, StrongExtractor(key))
        kw_r = f"{kw['r_phi']:.3f}" if not np.isnan(kw["r_phi"]) else "undefined"
        st_r = f"{st['r_phi']:.3f}" if not np.isnan(st["r_phi"]) else "undefined"
        print(f"{label:<28} {kw['n']:>6} {kw_r:>12} {st_r:>11}")
    print("-" * 72)
    print("Keyword r = lower bound (vocabulary tuned to stylized notes).")
    print("Strong r  = upper bound (oracle reader that perfectly interprets the")
    print("            note text; bounded by note<->condition decoupling noise).")
    print("These are SYNTHETIC simulation bounds, not an LLM and not real public text.")


if __name__ == "__main__":
    main()

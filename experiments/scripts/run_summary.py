"""Results summary (PROBLEM 5) — mean/std + paired Wilcoxon across seeds.

Produces, for each scenario:
  - mean and std of TRV, DCR, ICS across all seeds for every method
  - the paired Wilcoxon statistic and p-value (Ours vs Opt-only TRV)
  - a clearly labeled note on what the p-value does and does not measure

This is the canonical generator for the paper's Table IV. Numbers are whatever
the code produces — nothing is targeted. The same seed is used for every method
(shared asset population), so the Ours-vs-Opt comparison is correctly paired.

Usage:
    python scripts/run_summary.py --seeds 0-29
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.s2s.pipeline import load_config
from src.s2s.extractors.keyword import KeywordExtractor
from src.s2s.baselines.runner import run_baseline

METHODS = ["random", "rule_based", "xgboost", "opt_only", "semantic_only", "ours"]
SCENARIOS = [
    ("configs/s1_it.yaml", "S1: IT Infrastructure"),
    ("configs/s2_aviation.yaml", "S2: Aviation MRO"),
    ("configs/s3_consumer.yaml", "S3: Consumer Electronics"),
]

STABILITY_NOTE = (
    "p-value reflects simulation stability across seeds, "
    "not cross-dataset generalization"
)


def _parse_seeds(spec: str) -> list[int]:
    if "-" in spec and "," not in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in spec.split(",")]


def summarize_scenario(cfg_path: Path, label: str, seeds: list[int]):
    config = load_config(cfg_path)
    scenario_key = config["name"].split("_")[0]
    ext = KeywordExtractor(scenario_key)

    # Collect per-seed metrics for every method (same seeds -> paired).
    data = {m: {"TRV": [], "DCR": [], "ICS": []} for m in METHODS}
    for m in METHODS:
        for s in seeds:
            res = run_baseline(m, config, ext, s)
            data[m]["TRV"].append(res.TRV)
            data[m]["DCR"].append(res.DCR)
            data[m]["ICS"].append(res.ICS)

    print("=" * 78)
    print(f"{label}   (mean +/- std over {len(seeds)} seeds)")
    print("=" * 78)
    print(f"{'Method':<14} {'TRV mean':>13} {'TRV std':>11} {'DCR':>14} {'ICS mean':>12}")
    print("-" * 78)
    for m in METHODS:
        trv = np.array(data[m]["TRV"])
        dcr = np.array(data[m]["DCR"])
        ics = np.array(data[m]["ICS"])
        print(f"{m:<14} {trv.mean():>13,.0f} {trv.std():>11,.0f} "
              f"{dcr.mean():>7.1%}+/-{dcr.std():>4.1%} {ics.mean():>12,.0f}")

    # Paired Wilcoxon: Ours vs Opt-only on TRV (same seed per pair).
    ours = np.array(data["ours"]["TRV"])
    opt = np.array(data["opt_only"]["TRV"])
    diff = ours - opt
    lift_abs = ours.mean() - opt.mean()
    lift_pct = 100.0 * lift_abs / opt.mean() if opt.mean() else float("nan")
    print("-" * 78)
    print(f"Lift (Ours - Opt-only): {lift_abs:,.0f}  ({lift_pct:+.1f}%)")
    if np.allclose(diff, 0):
        print("Wilcoxon: undefined (all paired differences are zero)")
    else:
        w, p = stats.wilcoxon(ours, opt)
        print(f"Paired Wilcoxon (Ours vs Opt-only, n={len(seeds)}): "
              f"W={w:.1f}, p={p:.2e}")
    print(f"NOTE: {STABILITY_NOTE}.")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-29")
    args = ap.parse_args()
    seeds = _parse_seeds(args.seeds)
    base_dir = Path(__file__).parent.parent
    for cfg_path, label in SCENARIOS:
        summarize_scenario(base_dir / cfg_path, label, seeds)


if __name__ == "__main__":
    main()

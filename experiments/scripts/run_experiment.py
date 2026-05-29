"""CLI entry point: run experiment for one scenario across seeds and baselines.

Usage:
    python scripts/run_experiment.py --config configs/s1_it.yaml --seeds 0,1,2,3,4
    python scripts/run_experiment.py --config configs/s1_it.yaml --baseline ours --seeds 42
    python scripts/run_experiment.py --config configs/s1_it.yaml --capacity-sweep
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.s2s.pipeline import load_config, run_pipeline
from src.s2s.extractors.base import NullExtractor
from src.s2s.extractors.keyword import KeywordExtractor
from src.s2s.baselines.runner import run_baseline

BASELINES = ["random", "rule_based", "xgboost", "opt_only", "semantic_only", "ours"]
OUT_DIR = Path(__file__).parent.parent / "outputs"


def main():
    parser = argparse.ArgumentParser(description="Run S2S experiment")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seeds", default="0,1,2,3,4",
                       help="Comma-separated seeds")
    parser.add_argument("--baseline", default=None,
                       help="Run single baseline (default: all)")
    parser.add_argument("--capacity-sweep", action="store_true",
                       help="Run capacity sensitivity analysis")
    args = parser.parse_args()

    config = load_config(args.config)
    seeds = [int(s) for s in args.seeds.split(",")]
    scenario = config["name"]

    # Determine scenario key for keyword extractor
    scenario_key = scenario.split("_")[0]  # s1, s2, s3
    keyword_ext = KeywordExtractor(scenario_key)

    baselines = [args.baseline] if args.baseline else BASELINES
    OUT_DIR.mkdir(exist_ok=True)

    if args.capacity_sweep:
        _run_capacity_sweep(config, keyword_ext, seeds, scenario)
        return

    # Main experiment
    rows = []
    for bl in baselines:
        for seed in seeds:
            metrics = run_baseline(bl, config, keyword_ext, seed)
            row = {
                "scenario": scenario,
                "system": bl,
                "seed": seed,
                "TRV": metrics.TRV,
                "DCR": metrics.DCR,
                "ICS": metrics.ICS,
                "TPR": metrics.TPR,
            }
            rows.append(row)
            print(f"  {bl:<12} seed={seed} TRV=${metrics.TRV:>10,.0f} "
                  f"DCR={metrics.DCR:.0%} ICS=${metrics.ICS:>8,.0f}")

    # Write results
    out_path = OUT_DIR / f"{scenario}_results.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n-> {out_path}")

    # Summary
    print(f"\n{'System':<12} {'TRV mean':>12} {'TRV std':>10} {'DCR':>6} {'ICS':>10}")
    print("-" * 52)
    for bl in baselines:
        bl_rows = [r for r in rows if r["system"] == bl]
        trv_mean = np.mean([r["TRV"] for r in bl_rows])
        trv_std = np.std([r["TRV"] for r in bl_rows])
        dcr_mean = np.mean([r["DCR"] for r in bl_rows])
        ics_mean = np.mean([r["ICS"] for r in bl_rows])
        print(f"{bl:<12} ${trv_mean:>10,.0f} ${trv_std:>8,.0f} "
              f"{dcr_mean:>5.0%} ${ics_mean:>8,.0f}")


def _run_capacity_sweep(config, keyword_ext, seeds, scenario):
    """Capacity sensitivity: 50% to 150% in 10% steps."""
    levels = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    rows = []

    for cap in levels:
        for seed in seeds:
            metrics = run_baseline("ours", config, keyword_ext, seed,
                                  capacity_fraction=cap)
            rows.append({
                "scenario": scenario,
                "capacity_pct": cap,
                "seed": seed,
                "TRV": metrics.TRV,
            })

    out_path = OUT_DIR / f"{scenario}_capacity.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"-> {out_path}")

    # Normalize to 100% baseline
    baseline_trvs = [r["TRV"] for r in rows if r["capacity_pct"] == 1.0]
    baseline_mean = np.mean(baseline_trvs)

    print(f"\n{'Cap%':<8} {'Normalized':>10}")
    for cap in levels:
        cap_trvs = [r["TRV"] for r in rows if r["capacity_pct"] == cap]
        norm = np.mean(cap_trvs) / baseline_mean if baseline_mean else 1.0
        print(f"{cap:.0%}    {norm:>9.3f}")


if __name__ == "__main__":
    main()

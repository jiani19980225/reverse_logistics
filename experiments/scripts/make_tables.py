"""Generate Table 5 and Table 6 from output CSVs.

Reads per-seed results and produces:
  - outputs/table5_results.csv (aggregated mean ± std)
  - outputs/table6_ablation.csv
  - LaTeX table strings printed to stdout
"""

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.s2s.pipeline import load_config
from src.s2s.extractors.keyword import KeywordExtractor
from src.s2s.baselines.runner import run_baseline

OUT_DIR = Path(__file__).parent.parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

SEEDS = [0, 1, 2, 3, 4]
SCENARIOS = [
    ("configs/s1_it.yaml", "s1"),
    ("configs/s2_aviation.yaml", "s2"),
    ("configs/s3_consumer.yaml", "s3"),
]
BASELINES = ["random", "rule_based", "xgboost", "opt_only", "llm_only", "ours"]


def make_table5():
    print("=" * 60)
    print("TABLE 5: Recovery Performance Across Scenarios")
    print("=" * 60)

    rows = []
    base_dir = Path(__file__).parent.parent

    for cfg_path, scenario_key in SCENARIOS:
        config = load_config(base_dir / cfg_path)
        ext = KeywordExtractor(scenario_key)

        for bl in BASELINES:
            seed_metrics = []
            for seed in SEEDS:
                m = run_baseline(bl, config, ext, seed)
                seed_metrics.append(m)

            trv_mean = np.mean([m.TRV for m in seed_metrics])
            trv_std = np.std([m.TRV for m in seed_metrics])
            dcr_mean = np.mean([m.DCR for m in seed_metrics])
            ics_mean = np.mean([m.ICS for m in seed_metrics])

            rows.append({
                "scenario": config["name"], "system": bl,
                "TRV_mean": trv_mean, "TRV_std": trv_std,
                "DCR_mean": dcr_mean, "ICS_mean": ics_mean,
            })
            print(f"  {config['name']:<25} {bl:<12} "
                  f"TRV=${trv_mean:>10,.0f}±{trv_std:>6,.0f} "
                  f"DCR={dcr_mean:.0%} ICS=${ics_mean:>8,.0f}")

    out_path = OUT_DIR / "table5_results.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n-> {out_path}")


def make_table6():
    print("\n" + "=" * 60)
    print("TABLE 6: Ablation Study")
    print("=" * 60)

    base_dir = Path(__file__).parent.parent
    rows = []

    ablation_configs = [
        ("full_system", {"extractor": "keyword", "use_adaptive_inspection": True}),
        ("no_adaptive_insp", {"extractor": "keyword", "use_adaptive_inspection": False}),
        ("no_s2s_phi", {"extractor": "null", "use_adaptive_inspection": True}),
        ("no_both", {"extractor": "null", "use_adaptive_inspection": False}),
    ]

    for cfg_path, scenario_key in SCENARIOS:
        config = load_config(base_dir / cfg_path)

        for variant_name, settings in ablation_configs:
            from src.s2s.extractors.base import NullExtractor
            ext = KeywordExtractor(scenario_key) if settings["extractor"] == "keyword" else NullExtractor()

            seed_metrics = []
            for seed in SEEDS:
                from src.s2s.pipeline import run_pipeline
                m = run_pipeline(config, ext, seed,
                               use_adaptive_inspection=settings["use_adaptive_inspection"])
                seed_metrics.append(m)

            trv_mean = np.mean([m.TRV for m in seed_metrics])
            trv_std = np.std([m.TRV for m in seed_metrics])
            ics_mean = np.mean([m.ICS for m in seed_metrics])

            rows.append({
                "scenario": config["name"], "variant": variant_name,
                "TRV_mean": trv_mean, "TRV_std": trv_std, "ICS_mean": ics_mean,
            })
            print(f"  {config['name']:<25} {variant_name:<20} "
                  f"TRV=${trv_mean:>10,.0f}±{trv_std:>6,.0f}")

    out_path = OUT_DIR / "table6_ablation.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    make_table5()
    make_table6()

"""Generate Figure 2 from capacity sweep CSVs.

Reads outputs/{scenario}_capacity.csv files. If they don't exist,
runs the capacity sweep first.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.s2s.pipeline import load_config
from src.s2s.extractors.keyword import KeywordExtractor
from src.s2s.baselines.runner import run_baseline

OUT_DIR = Path(__file__).parent.parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

SEEDS = [0, 1, 2, 3, 4]
LEVELS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
SCENARIOS = [
    ("configs/s1_it.yaml", "s1", "S1: IT Infrastructure"),
    ("configs/s2_aviation.yaml", "s2", "S2: Aviation MRO"),
    ("configs/s3_consumer.yaml", "s3", "S3: Consumer Electronics"),
]


def make_figure2():
    print("Generating Figure 2...")
    base_dir = Path(__file__).parent.parent
    all_data = {}

    for cfg_path, scenario_key, label in SCENARIOS:
        config = load_config(base_dir / cfg_path)
        ext = KeywordExtractor(scenario_key)

        # Get baseline TRV at 100%
        baseline_trvs = [run_baseline("ours", config, ext, s, capacity_fraction=1.0).TRV
                        for s in SEEDS]
        baseline_mean = np.mean(baseline_trvs)

        normalized = {}
        for cap in LEVELS:
            trvs = [run_baseline("ours", config, ext, s, capacity_fraction=cap).TRV
                   for s in SEEDS]
            normalized[cap] = {
                "mean": np.mean(trvs) / baseline_mean * 100 if baseline_mean else 100,
                "std": np.std(trvs) / abs(baseline_mean) * 100 if baseline_mean else 0,
            }
        all_data[label] = normalized

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    styles = {
        "S1: IT Infrastructure": {"color": "#2196F3", "marker": "o"},
        "S2: Aviation MRO": {"color": "#F44336", "marker": "s"},
        "S3: Consumer Electronics": {"color": "#4CAF50", "marker": "^"},
    }

    caps_pct = [c * 100 for c in LEVELS]
    for label, data in all_data.items():
        means = [data[c]["mean"] for c in LEVELS]
        stds = [data[c]["std"] for c in LEVELS]
        s = styles[label]
        ax.plot(caps_pct, means, marker=s["marker"], color=s["color"],
                label=label, linewidth=1.5, markersize=6)
        ax.fill_between(caps_pct,
                       [m - sd for m, sd in zip(means, stds)],
                       [m + sd for m, sd in zip(means, stds)],
                       alpha=0.15, color=s["color"])

    ax.axhline(100, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axvline(100, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Processing Capacity (% of baseline)", fontsize=10)
    ax.set_ylabel("Recovery Value (% of baseline)", fontsize=10)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim(45, 155)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=9)

    plt.tight_layout()
    out_path = OUT_DIR / "figure2_capacity.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"-> {out_path}")

    # Also save raw data
    import csv
    csv_path = OUT_DIR / "figure2_capacity.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["capacity_pct", "scenario", "normalized_mean", "normalized_std"])
        for label, data in all_data.items():
            for cap in LEVELS:
                writer.writerow([cap, label, data[cap]["mean"], data[cap]["std"]])
    print(f"-> {csv_path}")


if __name__ == "__main__":
    make_figure2()

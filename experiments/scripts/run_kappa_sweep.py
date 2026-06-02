"""Kappa sensitivity sweep (paper Table VI) on S1.

PROBLEM 3 investigation. Sweeps kappa_base in {5,10,20,40,60,80} and reports
the TRV lift (Ours vs Opt-only) at each value, over the given seeds.

Wiring note: kappa_base IS connected to the yield model. s2s_update() sets
    alpha = phi*base_yield*kappa,  beta = (1-phi*base_yield)*kappa
so kappa_base scales alpha_c and beta_c (verified separately). However the Beta
MEAN, phi*base_yield, is invariant to kappa, and the decision engine ranks/values
assets by the Monte-Carlo MEAN of the yield distribution. Kappa therefore only
changes the variance (MC sampling noise) of the value estimate, not the expected
value that drives decisions. A near-flat sweep is the expected, honest result:
in this model the framework's value comes from WHICH assets it inspects, not from
how tightly the text signal concentrates the prior.

Usage:
    python scripts/run_kappa_sweep.py --seeds 0-29
"""

import argparse
import copy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.s2s.pipeline import load_config
from src.s2s.extractors.keyword import KeywordExtractor
from src.s2s.baselines.runner import run_baseline

KAPPAS = [5, 10, 20, 40, 60, 80]


def _parse_seeds(spec: str) -> list[int]:
    if "-" in spec and "," not in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in spec.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/s1_it.yaml")
    ap.add_argument("--seeds", default="0-29")
    args = ap.parse_args()
    seeds = _parse_seeds(args.seeds)
    base_dir = Path(__file__).parent.parent

    base_cfg = load_config(base_dir / args.config)
    scenario_key = base_cfg["name"].split("_")[0]
    ext = KeywordExtractor(scenario_key)

    print("=" * 60)
    print(f"KAPPA SENSITIVITY — {base_cfg['name']} ({len(seeds)} seeds)")
    print("=" * 60)
    print(f"{'kappa':>6} {'Ours TRV':>14} {'Opt TRV':>14} {'lift %':>8}")
    print("-" * 60)

    lifts = []
    for kb in KAPPAS:
        cfg = copy.deepcopy(base_cfg)
        cfg["kappa_base"] = kb  # inject swept value into the prior-update model
        ours = [run_baseline("ours", cfg, ext, s).TRV for s in seeds]
        opt = [run_baseline("opt_only", cfg, ext, s).TRV for s in seeds]
        ours_m, opt_m = np.mean(ours), np.mean(opt)
        lift = 100.0 * (ours_m - opt_m) / opt_m if opt_m else float("nan")
        lifts.append(lift)
        print(f"{kb:>6} {ours_m:>14,.0f} {opt_m:>14,.0f} {lift:>7.2f}%")

    print("-" * 60)
    print(f"Lift range across kappa: {min(lifts):.2f}% .. {max(lifts):.2f}% "
          f"(spread {max(lifts) - min(lifts):.2f} pp)")
    print("Flatness is expected: decisions use the Beta MEAN, which is "
          "invariant to kappa.")


if __name__ == "__main__":
    main()

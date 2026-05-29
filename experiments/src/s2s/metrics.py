"""Metrics — TRV, DCR, ICS, TPR as defined in Section IV-C."""

from dataclasses import dataclass
import numpy as np


@dataclass
class RunMetrics:
    TRV: float    # Total Recovery Value (net USD)
    DCR: float    # Demand Coverage Rate (0-1)
    ICS: float    # Inspection Cost Savings vs inspect-all (USD)
    TPR: float    # Throughput Rate (assets/hour)
    n_assets: int


def compute_metrics(results: list[dict], full_inspection_cost: float) -> RunMetrics:
    """Compute metrics from a list of per-asset result dicts.

    Each result dict must have:
        realized_value: float (revenue - processing cost)
        inspection_cost: float
        disposition: str ("l2_full", "l1_partial", "scrap", "refurb")
        time_min: float
    """
    n = len(results)
    if n == 0:
        return RunMetrics(TRV=0, DCR=0, ICS=0, TPR=0, n_assets=0)

    trv = sum(r["realized_value"] - r["inspection_cost"] for r in results)
    n_processed = sum(1 for r in results if r["disposition"] != "scrap")
    dcr = n_processed / n
    total_insp = sum(r["inspection_cost"] for r in results)
    ics = full_inspection_cost * n - total_insp
    total_time_hr = sum(r["time_min"] for r in results) / 60.0
    tpr = n / total_time_hr if total_time_hr > 0 else float("inf")

    return RunMetrics(TRV=trv, DCR=dcr, ICS=ics, TPR=tpr, n_assets=n)

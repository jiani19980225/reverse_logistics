"""Decision Engine — greedy Monte Carlo allocator under capacity constraints.

Evaluates Eq. 1 at each feasible disassembly level using N=1000 MC draws
from Beta distributions. Assets ranked by expected net value, allocated
greedily until capacity exhausted.
"""

import numpy as np
from ..s2s.beta_model import s2s_update, sample_yield, ground_truth_params


def compute_expected_value(
    components: dict,
    prices: dict,
    base_yields: dict,
    phi: float,
    sigma: float,
    kappa_base: float,
    gamma: float,
    rng: np.random.Generator,
    n_samples: int = 1000,
    refurb_value: float = 0.0,
) -> tuple[float, str]:
    """Expected recovery value via MC sampling with S2S-adjusted Beta.

    Evaluates both L0 (whole-unit refurb) and L2 (component harvest),
    returns whichever is higher.

    Returns (expected_value, best_level).
    """
    # L2: component-level harvest
    total = np.zeros(n_samples)
    for comp_name, count in components.items():
        if comp_name not in prices:
            continue
        price = prices[comp_name]
        base_y = base_yields.get(comp_name, 0.75)
        params = s2s_update(base_y, phi, sigma, kappa_base, gamma)
        yields = sample_yield(params, rng, n=n_samples)
        total += yields * price * count
    ev_l2 = float(total.mean())

    # L0: whole-unit refurb (if configured)
    # Refurb value scales with phi (condition factor)
    ev_l0 = refurb_value * phi if refurb_value > 0 else 0.0

    if ev_l0 > ev_l2:
        return ev_l0, "l0_refurb"
    return ev_l2, "l2"


def realize_value(
    components: dict,
    prices: dict,
    true_yield_factor: float,
    base_yields: dict,
    rng: np.random.Generator,
) -> float:
    """Draw realized value from ground-truth Beta (the 'reveal' step)."""
    total = 0.0
    for comp_name, count in components.items():
        if comp_name not in prices:
            continue
        price = prices[comp_name]
        base_y = base_yields.get(comp_name, 0.75)
        true_y = true_yield_factor * base_y
        params = ground_truth_params(min(true_y, 0.99))
        actual = sample_yield(params, rng, n=1)[0]
        total += actual * price * count
    return total


def greedy_allocate(
    assets: list[dict],
    capacity_minutes: float,
    prices: dict,
    base_yields: dict,
    kappa_base: float,
    gamma: float,
    processing_costs: dict,
    rng: np.random.Generator,
    disable_ranking: bool = False,
) -> list[dict]:
    """Greedy allocation: rank by EV, allocate until capacity exhausted.

    Each asset dict must have: components, phi, sigma, true_yield_factor,
    inspection (InspectionDecision), asset_type, age_bracket.

    Returns list of result dicts with: realized_value, inspection_cost,
    disposition, time_min.
    """
    # Compute expected values
    refurb_values = processing_costs.get("refurb_values", {})
    for a in assets:
        rv_refurb = refurb_values.get(a.get("asset_type", ""), 0.0)
        ev, best_level = compute_expected_value(
            a["components"], prices, base_yields,
            a["phi"], a["sigma"], kappa_base, gamma, rng,
            refurb_value=rv_refurb,
        )
        a["expected_value"] = ev
        a["best_level"] = best_level
        proc = processing_costs.get(best_level, processing_costs.get("l2", {}))
        a["proc_cost"] = proc.get("cost", 0)
        a["proc_time"] = proc.get("time_min", 0)
        a["total_time"] = a["inspection"].time_min + a["proc_time"]
        a["net_ev"] = a["expected_value"] - a["inspection"].cost - a["proc_cost"]

    # Sort by net expected value descending (or keep arrival order if disabled)
    if not disable_ranking:
        ranked = sorted(assets, key=lambda x: -x["net_ev"])
    else:
        ranked = assets  # process in arrival order (no optimization)

    used_minutes = 0.0
    results = []

    for a in ranked:
        if used_minutes + a["total_time"] > capacity_minutes:
            # Capacity exhausted — scrap
            results.append({
                "realized_value": 0.0,
                "inspection_cost": 0.0,
                "disposition": "scrap",
                "time_min": 0.0,
            })
            continue

        used_minutes += a["total_time"]

        # Realize value from ground truth
        if a.get("best_level") == "l0_refurb":
            # L0: whole-unit resale. Value depends on true condition.
            refurb_val = refurb_values.get(a.get("asset_type", ""), 0.0)
            rv = refurb_val * a["true_yield_factor"]  # condition-scaled
        else:
            # L2: component harvest
            rv = realize_value(
                a["components"], prices, a["true_yield_factor"],
                base_yields, rng
            )
        rv -= a["proc_cost"]

        # Rework cost: when inspection was skipped (level=0) and asset is
        # actually bad (true_yield < 0.3), the system discovers mid-process
        # that the asset is unrecoverable. This incurs rework/scrap penalty:
        # wasted processing time + disposal cost + opportunity cost.
        # Only applies if scenario defines a rework cost (S3 consumer returns).
        rework_cost = 0.0
        if (a["inspection"].level == 0 and a["true_yield_factor"] < 0.30
                and "rework" in processing_costs):
            rework_cost = processing_costs["rework"]["cost"]
            rv -= rework_cost

        results.append({
            "realized_value": rv,
            "inspection_cost": a["inspection"].cost,
            "disposition": "l2_full" if a["net_ev"] > 0 else "scrap",
            "time_min": a["total_time"],
        })

    return results


def random_allocate(
    assets: list[dict],
    capacity_minutes: float,
    prices: dict,
    base_yields: dict,
    processing_costs: dict,
    rng: np.random.Generator,
) -> list[dict]:
    """Baseline 1: Random disposition — randomly assign L2/L1/scrap without optimization."""
    results = []
    used_minutes = 0.0

    # Shuffle order (random, not optimized)
    order = list(range(len(assets)))
    rng.shuffle(order)

    for idx in order:
        a = assets[idx]
        # Random disposition: 33% L2, 33% L1, 34% scrap
        roll = rng.random()
        if roll < 0.33:
            disp = "l2"
        elif roll < 0.66:
            disp = "l1"
        else:
            disp = "scrap"

        proc = processing_costs.get(disp, {})
        proc_cost = proc.get("cost", 0)
        proc_time = proc.get("time_min", 0)
        total_time = a["inspection"].time_min + proc_time

        if disp == "scrap" or used_minutes + total_time > capacity_minutes:
            results.append({"realized_value": 0.0, "inspection_cost": 0.0,
                          "disposition": "scrap", "time_min": 0.0})
            continue

        used_minutes += total_time
        # Realize value — random disposition means processing even bad assets
        rv = realize_value(a["components"], prices, a["true_yield_factor"],
                          base_yields, rng)
        if disp == "l1":
            rv *= 0.6  # L1 only recovers partial value
        rv -= proc_cost

        results.append({"realized_value": rv, "inspection_cost": a["inspection"].cost,
                       "disposition": disp, "time_min": total_time})
    return results


def fifo_allocate(
    assets: list[dict],
    capacity_minutes: float,
    prices: dict,
    base_yields: dict,
    processing_costs: dict,
    rng: np.random.Generator,
) -> list[dict]:
    """Baseline 2: FIFO rule-based — process ALL at L0 (whole-unit) in arrival order.

    Industry standard: no disassembly, no value optimization. Assets are
    processed as whole units (refurb/resell if functional, scrap if not).
    Recovery value is ~5-10% of component-level value because whole-unit
    resale captures only a fraction of embedded component value.
    """
    results = []
    used_minutes = 0.0
    proc = processing_costs.get("l1", {"cost": 20, "time_min": 10})

    for a in assets:
        total_time = a["inspection"].time_min + proc.get("time_min", 10)
        if used_minutes + total_time > capacity_minutes:
            results.append({"realized_value": 0.0, "inspection_cost": 0.0,
                          "disposition": "scrap", "time_min": 0.0})
            continue

        used_minutes += total_time
        # L0 whole-unit: only captures ~8% of component-level value
        # (whole server resale << sum of component values)
        rv = realize_value(a["components"], prices, a["true_yield_factor"],
                          base_yields, rng)
        rv *= 0.08  # L0 whole-unit recovery fraction
        rv -= proc.get("cost", 20)

        results.append({"realized_value": rv, "inspection_cost": a["inspection"].cost,
                       "disposition": "l0_whole", "time_min": total_time})
    return results


def threshold_allocate(
    assets: list[dict],
    capacity_minutes: float,
    prices: dict,
    base_yields: dict,
    processing_costs: dict,
    rng: np.random.Generator,
) -> list[dict]:
    """Baseline 5: Semantic-only threshold routing — no optimizer.

    Uses phi for disposition but lacks:
    1. Capacity-aware value ranking (processes in arrival order)
    2. Demand matching (recovered components may not match demand)
    3. Depth optimization (uses fixed phi thresholds, not EV-maximizing depth)

    Paper: "saves some inspection cost but performs poorly on recovery value
    because it lacks explicit capacity and demand allocation"
    """
    results = []
    used_minutes = 0.0

    for a in assets:
        phi = a["phi"]
        # Fixed threshold routing — no value-based depth optimization
        if phi > 0.7:
            disp = "l2"
        elif phi > 0.3:
            disp = "l1"
        else:
            disp = "scrap"

        proc = processing_costs.get(disp, {})
        proc_cost = proc.get("cost", 0)
        proc_time = proc.get("time_min", 0)
        total_time = a["inspection"].time_min + proc_time

        if disp == "scrap":
            results.append({"realized_value": 0.0, "inspection_cost": a["inspection"].cost,
                          "disposition": "scrap", "time_min": a["inspection"].time_min})
            used_minutes += a["inspection"].time_min
            continue

        if used_minutes + total_time > capacity_minutes:
            results.append({"realized_value": 0.0, "inspection_cost": 0.0,
                          "disposition": "scrap", "time_min": 0.0})
            continue

        used_minutes += total_time
        rv = realize_value(a["components"], prices, a["true_yield_factor"],
                          base_yields, rng)
        if disp == "l1":
            rv *= 0.6  # L1 partial recovery

        # No demand matching: only fraction of recovered value is realizable
        # because components don't align with active demand streams
        rv -= proc_cost

        results.append({"realized_value": rv, "inspection_cost": a["inspection"].cost,
                       "disposition": disp, "time_min": total_time})
    return results

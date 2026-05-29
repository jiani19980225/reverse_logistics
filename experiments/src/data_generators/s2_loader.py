"""S2 Loader — Aviation MRO asset generation.

In production: parses real FAA SDR CSV (191K records).
For simulation: generates synthetic SDR-style records calibrated to
the real SDR condition distribution (corroded 20%, cracked 17%, etc.).
"""

import numpy as np

_SDR_TEMPLATES = {
    "corroded": [
        "Corroded skin panel at STA {sta}. Fastener holes show pitting.",
        "Corrosion found on wing spar cap during C-check. Beyond blend limits.",
        "Floor beam corroded at lavatory area. Exfoliation type corrosion.",
    ],
    "cracked": [
        "Cracked window frame at station {sta}. Fatigue crack 2.5 inches.",
        "Crack found in engine mount during borescope. Exceeds serviceable limits.",
        "Pressure bulkhead crack detected during NDI. Requires structural repair.",
    ],
    "inoperative": [
        "Nav unit inoperative. No output on test bench. Suspected circuit board failure.",
        "Comm radio intermittent. Fails self-test on channel 3.",
    ],
    "failed": [
        "Turbine blade failed. Metal contamination in oil filter at {hours} hrs.",
        "Bearing seized during ground run. Engine removed for shop visit.",
    ],
    "worn": [
        "Brake assembly worn beyond limits at {hours} landings. Requires overhaul.",
        "Flight control cable worn. Strand breakage exceeds allowable per AMM.",
        "Seat track rollers worn. Binding noted during adjustment.",
    ],
    "serviceable": [
        "Component serviceable. Within limits per AMM. No defect found.",
        "Overhauled per SB-{sta}. All measurements within tolerance.",
        "Routine removal at {hours} flight hours. Complies with AD requirements.",
        "Repaired per SRM. Returned to serviceable condition.",
    ],
}

_CONDITION_DIST = {
    "corroded": 0.20, "cracked": 0.17, "inoperative": 0.07,
    "failed": 0.03, "worn": 0.25, "serviceable": 0.28,
}

_YIELD_MAP = {
    "corroded": (0.20, 0.50),
    "cracked": (0.10, 0.40),
    "inoperative": (0.30, 0.60),
    "failed": (0.05, 0.25),
    "worn": (0.40, 0.70),
    "serviceable": (0.75, 0.95),
}


def generate_s2_assets(config: dict, rng: np.random.Generator) -> list[dict]:
    n = config["n_assets"]
    types = config["asset_types"]
    type_weights = np.array([t["weight"] for t in types])
    type_weights /= type_weights.sum()

    conditions = list(_CONDITION_DIST.keys())
    cond_probs = np.array([_CONDITION_DIST[c] for c in conditions])
    cond_probs /= cond_probs.sum()

    assets = []
    for i in range(n):
        atype = types[rng.choice(len(types), p=type_weights)]
        age_bracket = int(rng.choice([0, 1], p=[0.5, 0.5]))

        condition = conditions[rng.choice(len(conditions), p=cond_probs)]
        templates = _SDR_TEMPLATES[condition]
        note = templates[rng.integers(0, len(templates))]
        note = note.format(sta=int(rng.integers(100, 999)),
                          hours=int(rng.integers(5000, 25000)))

        yf_lo, yf_hi = _YIELD_MAP[condition]
        true_yield = float(rng.uniform(yf_lo, yf_hi))

        assets.append({
            "asset_id": i,
            "asset_type": atype["name"],
            "age_bracket": age_bracket,
            "components": dict(atype["components"]),
            "text": note,
            "true_yield_factor": true_yield,
        })

    return assets

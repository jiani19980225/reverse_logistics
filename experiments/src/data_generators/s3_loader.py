"""S3 Loader — Consumer Electronics asset generation.

In production: parses CPSC SaferProducts.gov + Amazon Reviews.
For simulation: generates synthetic consumer return descriptions
calibrated to CPSC remedy distributions.
"""

import numpy as np

_TEMPLATES = {
    "functional_return": [
        "Wrong item shipped. Unopened box. Like new condition.",
        "Changed mind. Works fine, just don't need it anymore.",
        "Bought two by accident. This one never opened.",
        "Gift recipient already had one. Unused, sealed.",
    ],
    "cosmetic": [
        "Cosmetic only - small scratch on back. Fully functional.",
        "Minor dent on corner from shipping. Everything works.",
        "Screen has one dead pixel in corner. Otherwise perfect.",
    ],
    "degraded": [
        "Battery drains fast. Otherwise works. {age} months old.",
        "Slow performance after update. Might need factory reset.",
        "Charging port loose. Have to hold cable at angle.",
        "Speaker crackles at high volume. Rest is fine.",
    ],
    "dead": [
        "Dead. Won't turn on at all. Tried everything.",
        "It just stopped working after a week.",
        "Screen went black randomly. No response to any button.",
        "Sometimes it works sometimes it doesn't. Mostly doesn't now.",
    ],
    "hazard": [
        "Swollen battery. Phone is bulging. Scared to use it.",
        "Smoke came out when charging. Stopped using immediately.",
        "Got very hot while charging. Burn mark on table.",
    ],
}

_CONDITION_DIST = {
    "functional_return": 0.25,
    "cosmetic": 0.15,
    "degraded": 0.30,
    "dead": 0.20,
    "hazard": 0.10,
}

_YIELD_MAP = {
    "functional_return": (0.90, 0.99),  # refurbishable
    "cosmetic": (0.75, 0.90),
    "degraded": (0.40, 0.65),
    "dead": (0.10, 0.30),
    "hazard": (0.02, 0.10),
}


def generate_s3_assets(config: dict, rng: np.random.Generator) -> list[dict]:
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
        age_bracket = 0  # consumer electronics are typically young

        condition = conditions[rng.choice(len(conditions), p=cond_probs)]
        templates = _TEMPLATES[condition]
        note = templates[rng.integers(0, len(templates))]
        note = note.format(age=int(rng.integers(1, 24)))

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

"""S3 Loader — Consumer Electronics asset generation.

In production: parses CPSC SaferProducts.gov + Amazon Reviews.
For simulation: generates synthetic consumer return descriptions
calibrated to CPSC remedy distributions.

DECOUPLED GENERATION (PROBLEM 1 fix, extended to S3): true_yield is drawn from
the ground-truth condition; the review text is generated from an independently
sampled NOISY observation of that condition (omission, severity mislabel). The
calibration r therefore reflects genuine signal recovery, not construction.
"""

import numpy as np
from .noise import observed_condition, resolve_noise

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
    # Uninformative reviews used only for the "omission" noise mode (signal lost).
    # Deliberately free of any keyword-vocabulary signal.
    "uninformative": [
        "Returned item. No reason provided.",
        "Customer return processed. No description given.",
        "Item returned within the window. Condition not specified.",
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

# Severity-ordered (best -> worst recovery) for mislabel noise.
_SEVERITY_ORDER = ["functional_return", "cosmetic", "degraded", "dead", "hazard"]


def generate_s3_assets(config: dict, rng: np.random.Generator) -> list[dict]:
    n = config["n_assets"]
    types = config["asset_types"]
    type_weights = np.array([t["weight"] for t in types])
    type_weights /= type_weights.sum()

    conditions = list(_CONDITION_DIST.keys())
    cond_probs = np.array([_CONDITION_DIST[c] for c in conditions])
    cond_probs /= cond_probs.sum()

    p_omit, p_mislabel = resolve_noise(config)

    assets = []
    for i in range(n):
        atype = types[rng.choice(len(types), p=type_weights)]
        age_bracket = 0  # consumer electronics are typically young

        # Ground-truth condition -> true_yield (independent of the review text).
        condition = conditions[rng.choice(len(conditions), p=cond_probs)]
        yf_lo, yf_hi = _YIELD_MAP[condition]
        true_yield = float(rng.uniform(yf_lo, yf_hi))

        # Review text generated from a NOISY observation of the condition.
        observed = observed_condition(condition, _SEVERITY_ORDER, rng,
                                      p_omit, p_mislabel, omit_label="uninformative")
        templates = _TEMPLATES[observed]
        note = templates[rng.integers(0, len(templates))]
        note = note.format(age=int(rng.integers(1, 24)))

        assets.append({
            "asset_id": i,
            "asset_type": atype["name"],
            "age_bracket": age_bracket,
            "components": dict(atype["components"]),
            "text": note,
            "true_yield_factor": true_yield,
        })

    return assets

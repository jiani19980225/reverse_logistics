"""S1 Generator — IT Infrastructure asset generation.

Generates 500 synthetic technician notes with controlled signal distribution.
Vocabulary drawn from iFixit teardown reports. Condition severity from
categorical distribution calibrated to industry decommission statistics.
"""

import numpy as np

_TEMPLATES = {
    "clean": [
        "Routine decommission. All components seated properly. No corrosion. {age}yr service.",
        "Standard lifecycle replacement. Clean internals, no damage. Passed diagnostics.",
        "Scheduled refresh. Normal wear only. All slots populated, no bent pins.",
        "End of lease return. Functional, clean. No issues noted during final check.",
    ],
    "mixed": [
        "Fan loud on startup but functional. Minor dust buildup. {age}yr old.",
        "One DIMM slot shows intermittent errors. Rest of system clean.",
        "Minor scratch on chassis. Occasional thermal throttling under load.",
        "PSU fan noisy. Otherwise functional. Some cable wear near power connector.",
        "Slight oxidization on rear I/O panel. All ports tested functional.",
    ],
    "damaged": [
        "PSU failure. Visible burn marks on mainboard near power connector J12. CPU smells burnt.",
        "Water damage near DIMM slots A1-A4. Corrosion on contacts. Short circuit suspected.",
        "Thermal damage to GPU. Swollen capacitors on VRM. Bent PCIe slot.",
        "Multiple bent pins on CPU socket. Board flexion damage from shipping.",
        "Smoke damage. Melted plastic near power supply. Do not power on.",
    ],
    "ambiguous": [
        "Decommissioned. No notes from previous tech.",
        "Pulled from rack B7. Status unknown. Needs verification.",
        "Asset tag mismatch. Physical condition not assessed.",
    ],
}

_YIELD_FACTORS = {
    "clean": (0.85, 0.99),
    "mixed": (0.45, 0.75),
    "damaged": (0.05, 0.30),
    "ambiguous": (0.30, 0.70),
}


def generate_s1_assets(config: dict, rng: np.random.Generator) -> list[dict]:
    n = config["n_assets"]
    types = config["asset_types"]
    type_weights = np.array([t["weight"] for t in types])
    type_weights /= type_weights.sum()

    note_dist = config.get("note_distribution",
                           {"clean": 0.20, "mixed": 0.40, "damaged": 0.30, "ambiguous": 0.10})
    conditions = list(note_dist.keys())
    cond_probs = np.array([note_dist[c] for c in conditions])
    cond_probs /= cond_probs.sum()

    assets = []
    for i in range(n):
        atype = types[rng.choice(len(types), p=type_weights)]
        age_bracket = int(rng.choice([0, 1], p=[0.6, 0.4]))
        age_years = int(rng.integers(0, 3) if age_bracket == 0 else rng.integers(3, 6))

        condition = conditions[rng.choice(len(conditions), p=cond_probs)]
        templates = _TEMPLATES[condition]
        note = templates[rng.integers(0, len(templates))]
        note = note.format(age=age_years)

        yf_lo, yf_hi = _YIELD_FACTORS[condition]
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

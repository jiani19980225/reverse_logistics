"""Strong (oracle-text) extractor — an UPPER-BOUND reference for calibration.

This is NOT an LLM and NOT a real-data extractor. It is a deterministic
reference reader that correctly interprets the descriptive language in a note
and maps it to a recoverability point estimate via a table declared before
extraction. It represents the ceiling an ideal text reader could reach on the
*synthetic* benchmark, which is bounded by the irreducible note<->condition
decoupling noise.

Paired with the brittle KeywordExtractor it brackets achievable calibration:
  - KeywordExtractor  -> lower bound (vocabulary tuned to stylized notes)
  - StrongExtractor    -> upper bound (perfect interpretation of the note text)

It exists to support the paper's "value scales with extractor quality" claim
honestly and reproducibly, replacing the unsubstantiated real-text/LLM r=0.88
numbers that no released code or data could reproduce.
"""

import numpy as np
from .base import AbstractExtractor, ExtractionResult

# Distinctive phrases per (scenario, condition). Chosen to avoid cross-condition
# collisions (e.g. "no corrosion" in a clean note must NOT match "corrosion on").
_PATTERNS = {
    "s1": {
        "clean": ["routine decommission", "seated properly", "no corrosion",
                  "standard lifecycle", "clean internals", "passed diagnostics",
                  "scheduled refresh", "normal wear", "end of lease", "no issues noted"],
        "mixed": ["fan loud", "minor dust", "intermittent error", "minor scratch",
                  "thermal throttling", "psu fan noisy", "cable wear", "oxidiz"],
        "damaged": ["psu failure", "burn mark", "smells burnt", "water damage",
                    "short circuit", "thermal damage", "swollen capacitor",
                    "bent pcie", "bent pin", "board flexion", "smoke damage",
                    "melted plastic", "do not power on"],
        "ambiguous": ["no notes from previous", "status unknown", "needs verification",
                      "asset tag mismatch", "condition not assessed"],
    },
    "s2": {
        "serviceable": ["serviceable", "within limits", "no defect found",
                        "within tolerance", "complies with ad", "returned to serviceable"],
        "worn": ["worn beyond limits", "cable worn", "rollers worn", "requires overhaul"],
        "inoperative": ["inoperative", "fails self-test", "intermittent"],
        "corroded": ["corroded", "corrosion", "pitting", "exfoliation"],
        "cracked": ["cracked", "crack found", "fatigue crack", "bulkhead crack"],
        "failed": ["failed", "metal contamination", "seized"],
    },
    "s3": {
        "functional_return": ["wrong item", "changed mind", "never opened",
                              "unused", "sealed", "like new", "don't need it"],
        "cosmetic": ["cosmetic only", "small scratch", "minor dent", "dead pixel"],
        "degraded": ["battery drains", "slow performance", "charging port loose",
                     "speaker crackles", "factory reset"],
        "dead": ["won't turn on", "stopped working", "screen went black",
                 "sometimes it works", "dead."],
        "hazard": ["swollen battery", "bulging", "smoke came out", "very hot",
                   "burn mark"],
    },
}

# Recoverability map (declared before extraction): condition -> phi point estimate.
_PHI = {
    "s1": {"clean": 0.92, "mixed": 0.60, "damaged": 0.18, "ambiguous": 0.50},
    "s2": {"serviceable": 0.85, "worn": 0.55, "inoperative": 0.45,
           "corroded": 0.35, "cracked": 0.25, "failed": 0.15},
    "s3": {"functional_return": 0.95, "cosmetic": 0.82, "degraded": 0.52,
           "dead": 0.20, "hazard": 0.06},
}


class StrongExtractor(AbstractExtractor):
    """Oracle-text reader: classify the note's described condition, map to phi."""

    def __init__(self, scenario: str):
        if scenario not in _PATTERNS:
            raise ValueError(f"Unknown scenario: {scenario}")
        self.scenario = scenario
        self.patterns = _PATTERNS[scenario]
        self.phi_map = _PHI[scenario]

    def extract(self, text: str, rng: np.random.Generator,
                asset: dict = None) -> ExtractionResult:
        t = text.lower()
        # Score each condition by number of distinctive phrases present.
        scores = {cond: sum(p in t for p in phrases)
                  for cond, phrases in self.patterns.items()}
        best = max(scores, key=scores.get)
        if scores[best] == 0:
            # Uninformative note (omission): no signal recovered -> fallback prior.
            return ExtractionResult(phi=1.0, sigma=0.20)
        return ExtractionResult(phi=float(self.phi_map[best]), sigma=0.90)

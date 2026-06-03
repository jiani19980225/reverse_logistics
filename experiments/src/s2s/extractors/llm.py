"""LLM extractor — optional Anthropic Claude implementation of the extractor interface.

This is the concrete, runnable backing for the paper's "extractor-agnostic" claim:
it satisfies the same AbstractExtractor contract as the keyword and full-vocabulary
extractors, returns a context factor phi in (0,1] and confidence sigma in [0,1],
and plugs into the pipeline with no changes to the decision engine.

It is OPTIONAL and NOT part of the reproducible core:
  - Requires the `anthropic` package (pip install -r requirements-llm.txt) and an
    ANTHROPIC_API_KEY. Without either, construction raises a clear error.
  - LLM output is not bit-reproducible the way the deterministic extractors are,
    so it is excluded from the seeded simulation. Use it for the held-out
    calibration study (scripts/run_calibration.py --llm), not the main tables.
  - A response cache (dict) can be supplied so repeated runs over the same notes
    return identical results and do not re-pay API cost.

Design follows the Anthropic Python SDK guidance: structured outputs via
output_config.format, prompt caching on the stable per-scenario system prompt,
and claude-opus-4-8 as the default model.
"""

from __future__ import annotations

import json
import os

import numpy as np

from .base import AbstractExtractor, ExtractionResult

_DEFAULT_MODEL = "claude-opus-4-8"

# Per-scenario recoverability framing. Declared BEFORE extraction (the same
# protocol the deterministic extractors follow): the model reads only the note
# text and maps the described condition to a yield context factor.
_GUIDANCE = {
    "s1": (
        "You assess decommissioned IT hardware (servers, drives, memory, GPUs) "
        "from a technician's note. Map the described physical condition to a yield "
        "context factor phi in (0,1]: ~0.9-1.0 for clean/routine units with no damage; "
        "~0.5-0.7 for mixed signals (minor wear, intermittent faults); ~0.05-0.3 for "
        "clear damage (burn marks, water damage, corrosion, bent pins, swollen caps); "
        "~0.5 when the note is uninformative. phi must never exceed 1.0."
    ),
    "s2": (
        "You assess aviation components from an FAA Service Difficulty Report-style "
        "maintenance note. Map the described condition to a yield context factor phi "
        "in (0,1]: ~0.85 serviceable; ~0.55 worn; ~0.45 inoperative; ~0.35 corroded; "
        "~0.25 cracked; ~0.15 failed; ~0.5 when uninformative. A logged repair does not "
        "guarantee the root cause was resolved. phi must never exceed 1.0."
    ),
    "s3": (
        "You assess consumer-electronics returns from a customer description. Map the "
        "described condition to a yield context factor phi in (0,1]: ~0.9-0.99 functional/"
        "unopened returns; ~0.8 cosmetic-only damage; ~0.5 degraded (battery, charging, "
        "performance); ~0.2 dead; ~0.06 safety hazard (swollen battery, smoke); ~0.5 when "
        "uninformative. phi must never exceed 1.0."
    ),
}

_SYSTEM_TEMPLATE = (
    "{guidance}\n\n"
    "Also return sigma in [0,1], your confidence that the note contains enough "
    "information to assess condition: high (~0.85) for explicit, descriptive notes; "
    "low (~0.2) for vague, uninformative, or missing detail. Confidence and condition "
    "are independent: you can be highly confident a unit is damaged (low phi, high sigma). "
    "Respond only with the structured fields."
)

# Structured-output schema. Numerical min/max constraints are not supported by
# structured outputs, so phi/sigma are clipped client-side after parsing.
_SCHEMA = {
    "type": "object",
    "properties": {
        "phi": {"type": "number", "description": "yield context factor in (0,1]"},
        "sigma": {"type": "number", "description": "confidence in [0,1]"},
        "condition": {"type": "string", "description": "one-word condition label"},
    },
    "required": ["phi", "sigma", "condition"],
    "additionalProperties": False,
}


class LLMExtractor(AbstractExtractor):
    """Anthropic Claude extractor. Reads a note, returns ExtractionResult(phi, sigma).

    Args:
        scenario: "s1", "s2", or "s3" (selects the recoverability framing).
        model: Claude model ID (default claude-opus-4-8; pass "claude-haiku-4-5"
            to trade some quality for much lower cost on large note sets).
        client: a pre-built Anthropic client, or an object exposing the same
            `.messages.create(...)` surface. Injected in tests so no network or
            `anthropic` package is required. If None, a client is built lazily
            on first use and ANTHROPIC_API_KEY must be set.
        response_cache: optional dict mapping note text -> (phi, sigma). Populated
            as notes are scored; pass a persisted dict for reproducible, no-cost
            re-runs over the same corpus.
    """

    def __init__(self, scenario: str, model: str = _DEFAULT_MODEL,
                 client=None, response_cache: dict = None):
        if scenario not in _GUIDANCE:
            raise ValueError(f"Unknown scenario: {scenario}")
        self.scenario = scenario
        self.model = model
        self._client = client
        self._cache = response_cache if response_cache is not None else {}
        self._system = _SYSTEM_TEMPLATE.format(guidance=_GUIDANCE[scenario])

    def _client_or_build(self):
        """Return the injected client, or build one. Fail loudly if unavailable."""
        if self._client is not None:
            return self._client
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError(
                "LLMExtractor needs the anthropic package. "
                "Install it with: pip install -r requirements-llm.txt"
            ) from e
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "LLMExtractor needs ANTHROPIC_API_KEY in the environment. "
                "Set it, or inject a client for offline use."
            )
        self._client = anthropic.Anthropic()
        return self._client

    def extract(self, text: str, rng: np.random.Generator,
                asset: dict = None) -> ExtractionResult:
        if text in self._cache:
            phi, sigma = self._cache[text]
            return ExtractionResult(phi=phi, sigma=sigma)

        client = self._client_or_build()
        # Prompt caching: the per-scenario system prompt is stable across every
        # note, so cache it. (For a short system prompt the prefix may fall below
        # the model's minimum cacheable size, in which case caching is a no-op.)
        resp = client.messages.create(
            model=self.model,
            max_tokens=512,
            system=[{
                "type": "text",
                "text": self._system,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": f"Note:\n{text}"}],
            output_config={"format": {"type": "json_schema", "schema": _SCHEMA}},
        )
        raw = next(b.text for b in resp.content if b.type == "text")
        data = json.loads(raw)

        phi = float(np.clip(data["phi"], 0.01, 1.0))
        sigma = float(np.clip(data["sigma"], 0.0, 1.0))
        self._cache[text] = (phi, sigma)
        return ExtractionResult(phi=phi, sigma=sigma)

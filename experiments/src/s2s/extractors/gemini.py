"""Gemini extractor — optional Google Gemini implementation of the extractor interface.

A provider-neutral sibling of LLMExtractor (Anthropic): same AbstractExtractor
contract, returns phi in (0,1] and sigma in [0,1], reuses the identical
per-scenario recoverability framing so the two are directly comparable.

OPTIONAL and NOT part of the reproducible core:
  - Requires the `google-genai` package (pip install -r requirements-llm.txt) and
    a GEMINI_API_KEY (or GOOGLE_API_KEY). Without either, construction-time use
    raises a clear error.
  - LLM output is not bit-reproducible, so it is excluded from the seeded
    simulation. Use it for the held-out calibration study
    (scripts/run_calibration.py --llm --llm-provider gemini).
  - Supply a response_cache dict for reproducible, no-cost re-runs.

Uses the `google-genai` SDK: client.models.generate_content with
response_mime_type="application/json". The JSON key contract is stated in the
system instruction (Gemini's JSON mode needs the shape spelled out).
"""

from __future__ import annotations

import json
import os

import numpy as np

from .base import AbstractExtractor, ExtractionResult
from .llm import _GUIDANCE, _SYSTEM_TEMPLATE  # shared per-scenario framing

_DEFAULT_MODEL = "gemini-2.5-flash"

# Gemini JSON mode needs the exact key shape named in the prompt.
_JSON_INSTRUCTION = (
    '\n\nReturn ONLY a JSON object with exactly these keys: '
    '"phi" (number, 0 to 1), "sigma" (number, 0 to 1), "condition" (string).'
)


class GeminiExtractor(AbstractExtractor):
    """Google Gemini extractor. Reads a note, returns ExtractionResult(phi, sigma).

    Args:
        scenario: "s1", "s2", or "s3" (selects the recoverability framing).
        model: Gemini model ID (default gemini-2.5-flash).
        client: a pre-built `google.genai` client, or any object exposing
            `.models.generate_content(...)`. Injected in tests so no network or
            `google-genai` package is required. If None, built lazily on first
            use and GEMINI_API_KEY (or GOOGLE_API_KEY) must be set.
        response_cache: optional dict mapping note text -> (phi, sigma) for
            reproducible, zero-cost re-runs over the same corpus.
    """

    def __init__(self, scenario: str, model: str = _DEFAULT_MODEL,
                 client=None, response_cache: dict = None):
        if scenario not in _GUIDANCE:
            raise ValueError(f"Unknown scenario: {scenario}")
        self.scenario = scenario
        self.model = model
        self._client = client
        self._cache = response_cache if response_cache is not None else {}
        self._system = _SYSTEM_TEMPLATE.format(guidance=_GUIDANCE[scenario]) + _JSON_INSTRUCTION

    def _client_or_build(self):
        """Return the injected client, or build one. Fail loudly if unavailable."""
        if self._client is not None:
            return self._client
        try:
            from google import genai
        except ImportError as e:
            raise RuntimeError(
                "GeminiExtractor needs the google-genai package. "
                "Install it with: pip install -r requirements-llm.txt"
            ) from e
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError(
                "GeminiExtractor needs GEMINI_API_KEY (or GOOGLE_API_KEY) in the "
                "environment. Set it, or inject a client for offline use."
            )
        self._client = genai.Client(api_key=key)
        return self._client

    def extract(self, text: str, rng: np.random.Generator,
                asset: dict = None) -> ExtractionResult:
        if text in self._cache:
            phi, sigma = self._cache[text]
            return ExtractionResult(phi=phi, sigma=sigma)

        client = self._client_or_build()
        resp = client.models.generate_content(
            model=self.model,
            contents=f"Note:\n{text}",
            config={
                "system_instruction": self._system,
                "response_mime_type": "application/json",
                "temperature": 0,  # steady classification across the corpus
            },
        )
        data = json.loads(resp.text)

        phi = float(np.clip(data["phi"], 0.01, 1.0))
        sigma = float(np.clip(data["sigma"], 0.0, 1.0))
        self._cache[text] = (phi, sigma)
        return ExtractionResult(phi=phi, sigma=sigma)

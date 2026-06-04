"""DeepSeek extractor — optional DeepSeek implementation via OpenAI-compatible API.

Same AbstractExtractor contract as the Anthropic and Gemini extractors.
DeepSeek exposes an OpenAI-compatible endpoint, so this uses the `openai`
package pointed at DeepSeek's base URL — no separate SDK needed.

OPTIONAL — not part of the reproducible core:
  - Requires: pip install openai (already in most environments)
  - Requires: DEEPSEEK_API_KEY environment variable
  - Response cache provided for reproducible, zero-cost re-runs.

Usage:
    python scripts/run_calibration.py --seeds 0-29 \\
        --llm --llm-provider deepseek --llm-sample 60
"""

from __future__ import annotations

import json
import os

import numpy as np

from .base import AbstractExtractor, ExtractionResult
from .llm import _GUIDANCE, _SYSTEM_TEMPLATE  # shared per-scenario framing

_DEFAULT_MODEL = "deepseek-chat"
_BASE_URL = "https://api.deepseek.com"


class DeepSeekExtractor(AbstractExtractor):
    """DeepSeek extractor via OpenAI-compatible API.

    Args:
        scenario:       "s1", "s2", or "s3".
        model:          DeepSeek model ID (default deepseek-chat).
        client:         injectable OpenAI client for offline tests.
        response_cache: dict mapping note text -> (phi, sigma); populated
                        on every live call and persisted by the calibration
                        script so re-runs are free.
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
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError(
                "DeepSeekExtractor needs the openai package. "
                "Install: pip install openai"
            ) from e
        key = os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            raise RuntimeError(
                "DeepSeekExtractor needs DEEPSEEK_API_KEY in the environment."
            )
        self._client = OpenAI(api_key=key, base_url=_BASE_URL)
        return self._client

    def extract(self, text: str, rng: np.random.Generator,
                asset: dict = None) -> ExtractionResult:
        if text in self._cache:
            phi, sigma = self._cache[text]
            return ExtractionResult(phi=phi, sigma=sigma)

        client = self._client_or_build()
        resp = client.chat.completions.create(
            model=self.model,
            temperature=0,          # deterministic classification
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": self._system + (
                    "\n\nReturn ONLY a JSON object with keys: "
                    "\"phi\" (number 0-1), \"sigma\" (number 0-1), "
                    "\"condition\" (string)."
                )},
                {"role": "user", "content": f"Note:\n{text}"},
            ],
        )
        data = json.loads(resp.choices[0].message.content)
        phi = float(np.clip(data["phi"], 0.01, 1.0))
        sigma = float(np.clip(data["sigma"], 0.0, 1.0))
        self._cache[text] = (phi, sigma)
        return ExtractionResult(phi=phi, sigma=sigma)

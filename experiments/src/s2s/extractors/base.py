"""Abstract extractor interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ExtractionResult:
    phi: float      # context factor in (0, 1]
    sigma: float    # confidence in [0, 1]


class AbstractExtractor(ABC):
    @abstractmethod
    def extract(self, text: str, rng: np.random.Generator,
                asset: dict = None) -> ExtractionResult:
        """Extract phi and sigma from asset text.

        `asset` is the full asset dict, passed so extractors that use structured
        features (e.g. the XGBoost proxy reading age_bracket) can access them.
        Text-only extractors ignore it.
        """
        ...


class NullExtractor(AbstractExtractor):
    """Returns phi=1.0, sigma=0.0 (no semantic information)."""

    def extract(self, text: str, rng: np.random.Generator,
                asset: dict = None) -> ExtractionResult:
        return ExtractionResult(phi=1.0, sigma=0.0)

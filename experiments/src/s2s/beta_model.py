"""S2S Beta Model — the mathematical core of the paper.

From Section III-A (revised):
    ỹ_c(m) = clip(φ(m) · ŷ_c, ε, 1-ε)   with ε = 10⁻³
    κ(σ) = κ_base · e^(γσ)               with κ_base=10, γ=1.0
    α'_c = ỹ_c(m) · κ(σ)
    β'_c = (1 - ỹ_c(m)) · κ(σ)

When σ→1: κ≈27.2 (tight distribution, high confidence)
When σ→0: κ≈10 (wide distribution, low confidence)
"""

import numpy as np
from dataclasses import dataclass

_EPS = 1e-3  # clipping bound to prevent degenerate Beta


@dataclass(frozen=True)
class BetaParams:
    alpha: float
    beta: float

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        s = self.alpha + self.beta
        return (self.alpha * self.beta) / (s * s * (s + 1))


def kappa_from_sigma(sigma: float, kappa_base: float = 10.0, gamma: float = 1.0) -> float:
    """Confidence-adaptive concentration: κ(σ) = κ_base · e^(γσ)."""
    return kappa_base * np.exp(gamma * sigma)


def s2s_update(
    base_yield: float,
    phi: float,
    sigma: float = 0.5,
    kappa_base: float = 10.0,
    gamma: float = 1.0,
) -> BetaParams:
    """Apply S2S context factor and confidence to yield prior.

    Args:
        base_yield: ŷ_c, baseline yield probability in (0, 1).
        phi: context factor from extractor, in (0, 1].
        sigma: extractor confidence in [0, 1].
        kappa_base: base concentration (default 10).
        gamma: exponential scaling factor (default 1.0).

    Returns:
        Updated BetaParams with α' and β'.
    """
    if not (0.0 < base_yield <= 1.0):
        raise ValueError(f"base_yield must be in (0, 1], got {base_yield}")
    if not (0.0 < phi <= 1.0):
        raise ValueError(f"phi must be in (0, 1], got {phi}")
    if not (0.0 <= sigma <= 1.0):
        raise ValueError(f"sigma must be in [0, 1], got {sigma}")

    # Clip adjusted mean to prevent degenerate distributions
    adjusted_yield = np.clip(phi * base_yield, _EPS, 1.0 - _EPS)

    # Confidence-adaptive concentration
    kappa = kappa_from_sigma(sigma, kappa_base, gamma)

    alpha = adjusted_yield * kappa
    beta = (1.0 - adjusted_yield) * kappa
    return BetaParams(alpha=alpha, beta=beta)


def sample_yield(params: BetaParams, rng: np.random.Generator, n: int = 1) -> np.ndarray:
    """Draw yield samples from the Beta distribution."""
    return rng.beta(params.alpha, params.beta, size=n)


def ground_truth_params(true_yield: float, kappa: float = 20.0) -> BetaParams:
    """Ground-truth Beta parameters for evaluation."""
    y = np.clip(true_yield, _EPS, 1.0 - _EPS)
    return BetaParams(alpha=y * kappa, beta=(1.0 - y) * kappa)

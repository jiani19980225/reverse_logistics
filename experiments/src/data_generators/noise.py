"""Decoupled note-generation noise (PROBLEM 1 fix, shared by S1/S2/S3).

The technician note / customer review is generated from a NOISY observation of
the ground-truth condition, never from the condition that sets true_yield. This
breaks the circular construction where text and yield came from the same label,
so any extractor<->yield correlation must be earned through genuine signal
recovery rather than baked in.
"""

import numpy as np

# Default corruption levels applied to every scenario unless overridden via
# config["note_noise"].
DEFAULT_NOTE_NOISE = {
    "p_omit": 0.15,      # observer omits detail -> vague/uninformative note
    "p_mislabel": 0.25,  # perceived severity off by one adjacent category
}


def observed_condition(
    true_condition: str,
    severity_order: list,
    rng: np.random.Generator,
    p_omit: float,
    p_mislabel: float,
    omit_label: str,
) -> str:
    """Return the condition the note *describes* — a noisy view of ground truth.

    Three realistic corruption modes:
      - omission : note becomes uninformative (`omit_label`), losing the signal
      - mislabel : perceived severity drifts to an adjacent category
      - faithful : note correctly reflects the true condition
                   (prob = 1 - p_omit - p_mislabel)
    """
    roll = rng.random()
    if roll < p_omit:
        return omit_label
    if roll < p_omit + p_mislabel and true_condition in severity_order:
        idx = severity_order.index(true_condition)
        shift = -1 if rng.random() < 0.5 else 1
        idx = int(np.clip(idx + shift, 0, len(severity_order) - 1))
        return severity_order[idx]
    return true_condition


def resolve_noise(config: dict) -> tuple:
    """(p_omit, p_mislabel) from config['note_noise'] merged over defaults."""
    noise = {**DEFAULT_NOTE_NOISE, **config.get("note_noise", {})}
    return noise["p_omit"], noise["p_mislabel"]

"""Discrete-time survival: yearly intervals, the logistic-hazard loss, risks, and Kaplan-Meier.

Interval j covers follow-up years [j, j+1). A death at exactly 5.0 years is in interval 5, so "death within
5 years" means time < 5 everywhere (risk_at, km_risk, the binary 5-year outcome).
"""

from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

N_INTERVALS = 10  # the longest follow-up in the public 2019 linkage is 113 months (9.4 years)


def discretise(time_years: np.ndarray) -> np.ndarray:
    """Interval index floor(time) of each person's death or censoring time."""
    time_years = np.asarray(time_years, dtype=float)
    if (time_years < 0).any() or (time_years >= N_INTERVALS).any():
        raise ValueError(f"follow-up times must lie in [0, {N_INTERVALS}) years")
    return np.floor(time_years).astype(int)


def logistic_hazard_loss(logits: torch.Tensor, interval: torch.Tensor, died: torch.Tensor) -> torch.Tensor:
    """Mean binary cross-entropy over the person-intervals at risk.

    logits (B, J) are the per-interval hazards on the logit scale. A death in interval j gives targets 0 for
    intervals 0..j-1 and 1 for interval j. A person censored in interval j survived intervals 0..j-1; the
    censoring interval itself is left out, because we don't know whether they survived all of it.
    """
    j = torch.arange(logits.shape[1], device=logits.device)[None, :]
    interval = interval[:, None]
    died = died[:, None].bool()
    at_risk = (j < interval) | ((j == interval) & died)
    if not at_risk.any():
        raise ValueError("no person-interval at risk in this batch")
    target = ((j == interval) & died).float()
    return F.binary_cross_entropy_with_logits(logits[at_risk], target[at_risk])


def risk_at(hazards: np.ndarray, years: int) -> np.ndarray:
    """Risk of death before `years`: 1 - prod_{j < years} (1 - h_j)."""
    return 1 - np.prod(1 - hazards[:, :years], axis=1)


def _km_curve(time: np.ndarray, event: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Event times and the Kaplan-Meier survival just after each of them."""
    time, event = np.asarray(time, dtype=float), np.asarray(event, dtype=bool)
    times = np.unique(time[event])
    at_risk = (time[None, :] >= times[:, None]).sum(axis=1)
    events = (time[None, :] == times[:, None])[:, event].sum(axis=1) if event.any() else np.zeros(0)
    return times, np.cumprod(1 - events / at_risk)


def _step(times: np.ndarray, surv: np.ndarray, t: np.ndarray, left: bool) -> np.ndarray:
    """Evaluate the KM step function at t: S(t) (right-continuous) or S(t-) with left=True."""
    idx = np.searchsorted(times, np.asarray(t, dtype=float), side="left" if left else "right")
    return np.r_[1.0, surv][idx]


def km_risk(time_years: np.ndarray, died: np.ndarray, years: float) -> float:
    """Kaplan-Meier risk of death before `years` (deaths at exactly `years` are not counted)."""
    times, surv = _km_curve(time_years, died)
    return float(1 - _step(times, surv, np.array([years]), left=True)[0])


def km_censoring(time_years: np.ndarray, died: np.ndarray) -> Callable[..., np.ndarray]:
    """Censoring survival G(t): Kaplan-Meier with the roles of death and censoring swapped (for IPCW)."""
    times, surv = _km_curve(time_years, 1 - np.asarray(died))

    def G(t: np.ndarray, left: bool = False) -> np.ndarray:
        return _step(times, surv, t, left)

    return G

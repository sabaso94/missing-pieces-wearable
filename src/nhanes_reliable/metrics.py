"""Survival metrics on the predicted 5-year risk, with person-level bootstrap CIs.

Every metric takes (time in years, died 0/1, predicted 5-year risk) and returns a float, NaN where it is
undefined (for example no deaths in a bootstrap sample). Each row is one person, so the bootstrap resamples
rows.
"""

import warnings
from typing import Callable

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index

from .survival import km_censoring, km_risk

HORIZON = 5.0
Metric = Callable[[np.ndarray, np.ndarray, np.ndarray], float]


def c_index(time: np.ndarray, died: np.ndarray, risk: np.ndarray) -> float:
    """Harrell's C: share of comparable pairs where the person who died first had the higher risk."""
    if np.sum(died) == 0:
        return float("nan")
    try:
        # lifelines expects a score where higher means longer survival, hence -risk.
        return float(concordance_index(time, -np.asarray(risk), died))
    except ZeroDivisionError:  # no comparable pairs
        return float("nan")


def uno_c(time: np.ndarray, died: np.ndarray, risk: np.ndarray, tau: float = HORIZON) -> float:
    """Uno's C truncated at tau: pairs weighted by 1 / G(T_i-)^2, which corrects for censoring."""
    time, died, risk = np.asarray(time, float), np.asarray(died, bool), np.asarray(risk, float)
    G = km_censoring(time, died)
    numerator = denominator = 0.0
    # Follow-up is in whole months, so group the deaths by time: everyone with a later time is comparable.
    for u in np.unique(time[died & (time < tau)]):
        later = np.sort(risk[time > u])  # j outlived the deaths at u
        if len(later) == 0:
            continue
        r = risk[died & (time == u)]
        below = np.searchsorted(later, r, side="left")  # j with lower risk: concordant
        ties = np.searchsorted(later, r, side="right") - below
        weight = 1 / G(np.array([u]), left=True)[0] ** 2
        numerator += weight * (below + 0.5 * ties).sum()
        denominator += weight * len(later) * len(r)
    return float(numerator / denominator) if denominator > 0 else float("nan")


def brier5(time: np.ndarray, died: np.ndarray, risk: np.ndarray) -> float:
    """IPCW Brier score at 5 years (Graf et al. 1999).

    Deaths before 5 y contribute (1 - risk)^2 / G(T_i-); people known alive before 5 y contribute
    risk^2 / G(5-); people censored before 5 y contribute 0 and are represented through the weights.
    """
    time, died, risk = np.asarray(time, float), np.asarray(died, bool), np.asarray(risk, float)
    G = km_censoring(time, died)
    dead = died & (time < HORIZON)
    alive = time >= HORIZON
    score = np.zeros(len(time))
    score[dead] = (1 - risk[dead]) ** 2 / G(time[dead], left=True)
    score[alive] = risk[alive] ** 2 / G(np.array([HORIZON]), left=True)[0]
    return float(score.mean())


def citl5(time: np.ndarray, died: np.ndarray, risk: np.ndarray) -> float:
    """Calibration-in-the-large: mean predicted 5-year risk minus the Kaplan-Meier observed risk.

    Positive means the model over-predicts. Reported in absolute risk (0.01 = 1 percentage point).
    """
    if len(time) == 0:
        return float("nan")
    return float(np.mean(risk) - km_risk(time, died, HORIZON))


def reliability_table(time: np.ndarray, died: np.ndarray, risk: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Mean predicted vs Kaplan-Meier observed 5-year risk per risk decile."""
    bins = pd.qcut(pd.Series(risk).rank(method="first"), n_bins, labels=False).to_numpy()
    rows = []
    for b in range(n_bins):
        idx = bins == b
        rows.append({"bin": b, "n": int(idx.sum()), "mean_pred": float(risk[idx].mean()),
                     "observed": km_risk(time[idx], died[idx], HORIZON)})
    return pd.DataFrame(rows)


def _percentiles(draws: np.ndarray) -> tuple[float, float]:
    """95% percentile interval over the finite draws (NaN draws, e.g. no deaths, are skipped)."""
    draws = np.asarray(draws, dtype=float)
    if not np.isfinite(draws).any():
        return float("nan"), float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def bootstrap_ci(metric: Metric, time: np.ndarray, died: np.ndarray, risk: np.ndarray,
                 n_boot: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """(point estimate, lower, upper) of a 95% percentile bootstrap CI, resampling people."""
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(time), len(time))
        draws.append(metric(time[idx], died[idx], risk[idx]))
    return (metric(time, died, risk), *_percentiles(draws))


def model_diff_ci(metric: Metric, time: np.ndarray, died: np.ndarray, risk_a: np.ndarray, risk_b: np.ndarray,
                  n_boot: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """Paired difference metric(a) - metric(b), both models scored on the same resampled people."""
    def diff(idx):
        return metric(time[idx], died[idx], risk_a[idx]) - metric(time[idx], died[idx], risk_b[idx])

    rng = np.random.default_rng(seed)
    draws = [diff(rng.integers(0, len(time), len(time))) for _ in range(n_boot)]
    return (diff(np.arange(len(time))), *_percentiles(draws))

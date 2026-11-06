"""Logistic recalibration of the predicted 5-year risk: logit(r') = a + b * logit(r).

Fitted on the binary outcome "died before 5 years" among people whose 5-year status is known (people censored
alive before 5 years are left out; 0.3% of the cohort). With b > 0 the map is monotone, so ranking (C-index)
within one fit is unchanged; only calibration moves.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from .metrics import HORIZON

EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_recal(time: np.ndarray, died: np.ndarray, risk: np.ndarray, years: float = HORIZON) -> np.ndarray:
    """[a, b] of the logistic recalibration, fitted by maximum likelihood (no penalty)."""
    known = (died == 1) | (time >= years)
    y = ((died == 1) & (time < years))[known].astype(int)
    model = LogisticRegression(penalty=None).fit(_logit(risk[known])[:, None], y)
    return np.array([model.intercept_[0], model.coef_[0, 0]])


def apply_recal(risk: np.ndarray, params: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-(params[0] + params[1] * _logit(risk))))


def fit_group_recal(time: np.ndarray, died: np.ndarray, risk: np.ndarray,
                    groups: np.ndarray) -> dict[str, np.ndarray]:
    """One recalibration per group (needs the group to be known when the model is used)."""
    groups = np.asarray(groups).astype(str)
    return {g: fit_recal(time[groups == g], died[groups == g], risk[groups == g]) for g in np.unique(groups)}


def apply_group_recal(risk: np.ndarray, groups: np.ndarray, params: dict[str, np.ndarray]) -> np.ndarray:
    groups = np.asarray(groups).astype(str)
    out = np.empty(len(risk))
    for g in np.unique(groups):
        if g not in params:
            raise KeyError(f"no recalibration fitted for group {g!r}; fitted groups: {sorted(params)}")
        out[groups == g] = apply_recal(risk[groups == g], params[g])
    return out

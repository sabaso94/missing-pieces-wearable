"""How much of an association with mortality could be an effect: risks under two exposures, adjusted.

aipw_risk estimates the risk of the outcome if everyone had been exposed and if no one had, by augmented inverse
probability weighting (doubly robust; Robins et al., 1994), with outcome and exposure models fitted by 5-fold
cross-fitting (Chernozhukov et al., 2018), linear or gradient-boosted (boosting stops early on a held-out 20% of
the training rows). Intervals come from the influence functions; the risk ratio's on the log scale. With n_splits > 1
the cross-fitting is repeated over random fold splits and the median is reported, with a standard error that adds
the spread between splits (Chernozhukov et al., 2018, section 3.4), so the result does not hinge on one split.
e_value is the minimum strength of association, on the risk-ratio scale, that an unmeasured confounder would need
with both exposure and outcome to explain an observed risk ratio away (VanderWeele and Ding, 2017).
"""
from typing import Callable

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

Z = 1.959963984540054


def _model(nuisance: str, seed: int):
    if nuisance == "linear":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=8, min_samples_leaf=20,
                                          early_stopping=True, validation_fraction=0.2, n_iter_no_change=10,
                                          random_state=seed)


def _fit_proba(make: Callable, x: np.ndarray, y: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    """P(y = 1) for x_test; the training mean if the training rows hold one class only."""
    if y.min() == y.max():
        return np.full(len(x_test), float(y.mean()))
    return make().fit(x, y).predict_proba(x_test)[:, 1]


def _summary(psi1: np.ndarray, psi0: np.ndarray) -> dict:
    n = len(psi1)
    r1, r0 = psi1.mean(), psi0.mean()
    se_rd = (psi1 - psi0).std(ddof=1) / np.sqrt(n)
    se_log_rr = ((psi1 - r1) / r1 - (psi0 - r0) / r0).std(ddof=1) / np.sqrt(n)
    rr = r1 / r0
    return {"risk1": float(r1), "risk0": float(r0), "rd": float(r1 - r0), "rd_low": float(r1 - r0 - Z * se_rd),
            "rd_high": float(r1 - r0 + Z * se_rd), "rr": float(rr), "rr_low": float(rr * np.exp(-Z * se_log_rr)),
            "rr_high": float(rr * np.exp(Z * se_log_rr))}


def risk_unadjusted(y: np.ndarray, t: np.ndarray) -> dict:
    """Risks in the exposed and unexposed as they are (no adjustment): the AIPW form with constant models."""
    p1, r1, r0 = t.mean(), y[t == 1].mean(), y[t == 0].mean()
    return _summary(r1 + t * (y - r1) / p1, r0 + (1 - t) * (y - r0) / (1 - p1))


def _aipw_once(y, t, x, nuisance, n_folds, seed, clip):
    folds = np.empty(len(y), dtype=int)
    folds[np.random.default_rng(seed).permutation(len(y))] = np.arange(len(y)) % n_folds
    mu1, mu0, e = np.empty(len(y)), np.empty(len(y)), np.empty(len(y))
    make = lambda: _model(nuisance, seed)
    for k in range(n_folds):
        test, train = folds == k, folds != k
        mu1[test] = _fit_proba(make, x[train & (t == 1)], y[train & (t == 1)], x[test])
        mu0[test] = _fit_proba(make, x[train & (t == 0)], y[train & (t == 0)], x[test])
        e[test] = _fit_proba(make, x[train], t[train], x[test])
    e = np.clip(e, clip, 1 - clip)
    psi1 = mu1 + t * (y - mu1) / e
    psi0 = mu0 + (1 - t) * (y - mu0) / (1 - e)
    return _summary(psi1, psi0), e


def _median_over_splits(outs: list[dict]) -> dict:
    """Median estimate over splits; SE = median of sqrt(se_s^2 + (est_s - median)^2), for RD and log RR."""
    rd = np.array([o["rd"] for o in outs])
    se_rd = np.array([(o["rd_high"] - o["rd_low"]) / (2 * Z) for o in outs])
    log_rr = np.log([o["rr"] for o in outs])
    se_log_rr = np.array([np.log(o["rr_high"] / o["rr_low"]) / (2 * Z) for o in outs])
    rd_m, lrr_m = np.median(rd), np.median(log_rr)
    se_rd_m = np.median(np.sqrt(se_rd ** 2 + (rd - rd_m) ** 2))
    se_lrr_m = np.median(np.sqrt(se_log_rr ** 2 + (log_rr - lrr_m) ** 2))
    return {"risk1": float(np.median([o["risk1"] for o in outs])), "risk0": float(np.median([o["risk0"] for o in outs])),
            "rd": float(rd_m), "rd_low": float(rd_m - Z * se_rd_m), "rd_high": float(rd_m + Z * se_rd_m),
            "rr": float(np.exp(lrr_m)), "rr_low": float(np.exp(lrr_m - Z * se_lrr_m)),
            "rr_high": float(np.exp(lrr_m + Z * se_lrr_m))}


def aipw_risk(y: np.ndarray, t: np.ndarray, x: np.ndarray, nuisance: str = "linear", n_folds: int = 5, seed: int = 0,
              clip: float = 0.01, return_propensity: bool = False, n_splits: int = 1):
    """Risks under exposure and under none, their difference and ratio with 95% CIs (and, if asked, the
    cross-fitted, clipped propensities of the first split). Split s uses seed + s."""
    runs = [_aipw_once(y, t, x, nuisance, n_folds, seed + s, clip) for s in range(n_splits)]
    out = runs[0][0] if n_splits == 1 else _median_over_splits([r[0] for r in runs])
    return (out, runs[0][1]) if return_propensity else out


def e_value(rr: float) -> float:
    rr = 1 / rr if rr < 1 else rr
    return float(rr + np.sqrt(rr * (rr - 1)))


def overlap(e: np.ndarray) -> dict:
    return {"below_0.05": float(np.mean(e < 0.05)), "above_0.95": float(np.mean(e > 0.95))}


def e_value_ci(rr: float, rr_low: float, rr_high: float) -> float:
    """E-value of the interval's bound nearer 1 (1 if the interval contains 1)."""
    if rr_low <= 1 <= rr_high:
        return 1.0
    return e_value(rr_high if rr < 1 else rr_low)

"""Subgroups: age bands, and per-group metrics with person-level bootstrap CIs on the gaps between groups."""

import numpy as np
import pandas as pd

from .metrics import HORIZON, Metric, _percentiles, bootstrap_ci, c_index, citl5
from .survival import km_risk

AGE_EDGES = (20, 50, 65)  # bands 20-49, 50-64, 65+ (NHANES top-codes age at 80)


def age_band(age: pd.Series, edges: tuple[int, ...] = AGE_EDGES) -> pd.Series:
    """Ordered categorical age band, labelled like "20-49" and "65+"."""
    labels = [f"{a}-{b - 1}" for a, b in zip(edges[:-1], edges[1:])] + [f"{edges[-1]}+"]
    return pd.cut(age, bins=list(edges) + [np.inf], right=False, labels=labels, ordered=True)


def subgroup_table(time: np.ndarray, died: np.ndarray, risk: np.ndarray, groups: pd.Series,
                   n_boot: int = 1000) -> pd.DataFrame:
    """Per group: size, deaths (all and within 5 years), observed and mean predicted 5-year risk, C-index and CITL with CIs."""
    groups = np.asarray(groups.astype(str))
    rows = []
    for g in pd.unique(groups):
        idx = groups == g
        t, d, r = time[idx], died[idx], risk[idx]
        row = {"group": g, "n": int(idx.sum()), "deaths": int(d.sum()),
               "deaths_5y": int(((d == 1) & (t < HORIZON)).sum()), "observed5": km_risk(t, d, HORIZON),
               "mean_pred": float(r.mean())}
        for name, metric in [("c_index", c_index), ("citl5", citl5)]:
            row[name], row[f"{name}_lo"], row[f"{name}_hi"] = bootstrap_ci(metric, t, d, r, n_boot)
        rows.append(row)
    return pd.DataFrame(rows)


def _gap(metric: Metric, time, died, risk, in_a, in_b) -> float:
    return metric(time[in_a], died[in_a], risk[in_a]) - metric(time[in_b], died[in_b], risk[in_b])


def gap_ci(metric: Metric, time: np.ndarray, died: np.ndarray, risk: np.ndarray, in_a: np.ndarray,
           in_b: np.ndarray, n_boot: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """metric(group a) - metric(group b), with a 95% CI from resampling people (both groups at once)."""
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(time), len(time))
        draws.append(_gap(metric, time[idx], died[idx], risk[idx], in_a[idx], in_b[idx]))
    return (_gap(metric, time, died, risk, in_a, in_b), *_percentiles(draws))


def gap_change_ci(metric: Metric, time: np.ndarray, died: np.ndarray, risk_new: np.ndarray, risk_old: np.ndarray,
                  in_a: np.ndarray, in_b: np.ndarray, n_boot: int = 1000,
                  seed: int = 0) -> tuple[float, float, float]:
    """Change of the a-b gap from the old to the new model, both scored on the same resampled people."""
    def change(t, d, new, old, a, b):
        return _gap(metric, t, d, new, a, b) - _gap(metric, t, d, old, a, b)

    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(time), len(time))
        draws.append(change(time[idx], died[idx], risk_new[idx], risk_old[idx], in_a[idx], in_b[idx]))
    return (change(time, died, risk_new, risk_old, in_a, in_b), *_percentiles(draws))

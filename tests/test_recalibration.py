import numpy as np
import pytest

from nhanes_reliable.metrics import c_index, citl5
from nhanes_reliable.recalibration import apply_group_recal, apply_recal, fit_group_recal, fit_recal


def toy(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    t = rng.exponential(1 / (0.04 * np.exp(x)))
    time, died = np.minimum(t, 8.0), (t < 8.0).astype(int)
    return time, died, 1 - np.exp(-0.04 * np.exp(x) * 5)


def logit(p):
    return np.log(p / (1 - p))


def test_recal_fixes_shifted_risk():
    time, died, risk = toy()
    inflated = 1 / (1 + np.exp(-(logit(risk) + 1.0)))
    assert citl5(time, died, inflated) > 0.05
    fixed = apply_recal(inflated, fit_recal(time, died, inflated))
    assert abs(citl5(time, died, fixed)) < 0.01


def test_recal_monotone_keeps_c_index():
    time, died, risk = toy()
    fixed = apply_recal(risk, fit_recal(time, died, risk))
    assert c_index(time, died, fixed) == pytest.approx(c_index(time, died, risk), abs=1e-12)


def test_group_recal_per_group():
    time, died, risk = toy()
    groups = np.where(np.arange(len(time)) % 2 == 0, "a", "b")
    shifted = np.where(groups == "a", 1 / (1 + np.exp(-(logit(risk) + 1.0))), risk)
    fixed = apply_group_recal(shifted, groups, fit_group_recal(time, died, shifted, groups))
    for g in ["a", "b"]:
        assert abs(citl5(time[groups == g], died[groups == g], fixed[groups == g])) < 0.015


def test_group_recal_missing_group_message():
    time, died, risk = toy()
    params = fit_group_recal(time, died, risk, np.full(len(time), "a"))
    with pytest.raises(KeyError, match="no recalibration fitted for group 'b'"):
        apply_group_recal(risk, np.full(len(time), "b"), params)

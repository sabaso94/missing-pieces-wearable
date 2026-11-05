import numpy as np
import pytest

from nhanes_reliable.metrics import (
    bootstrap_ci,
    brier5,
    c_index,
    citl5,
    model_diff_ci,
    reliability_table,
    uno_c,
)
from nhanes_reliable.survival import km_risk


def toy(n=400, seed=0):
    """Exponential survival whose hazard rises with x; administrative censoring at 8 years."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    t = rng.exponential(1 / (0.08 * np.exp(x)))
    time, died = np.minimum(t, 8.0), (t < 8.0).astype(int)
    return time, died, 1 - np.exp(-0.08 * np.exp(x) * 5)  # true 5-year risk


def test_c_index_perfect_and_reversed():
    time = np.array([1.0, 2.0, 3.0, 4.0])
    died = np.ones(4, dtype=int)
    risk = np.array([0.9, 0.7, 0.5, 0.1])  # earlier death, higher risk
    assert c_index(time, died, risk) == 1.0
    assert c_index(time, died, -risk) == 0.0
    assert c_index(time, died, np.full(4, 0.3)) == 0.5


def test_c_index_no_deaths_nan():
    assert np.isnan(c_index(np.array([1.0, 2.0]), np.array([0, 0]), np.array([0.1, 0.2])))


def test_uno_equals_harrell_without_censoring():
    rng = np.random.default_rng(1)
    time = rng.uniform(0, 4.9, 200)
    died = np.ones(200, dtype=int)
    risk = -time + rng.normal(0, 1, 200)
    assert uno_c(time, died, risk) == pytest.approx(c_index(time, died, risk), abs=1e-9)


def test_uno_reasonable_on_censored_data():
    time, died, risk = toy()
    assert 0.6 < uno_c(time, died, risk) < 0.85


def test_brier5_no_censoring_is_mse():
    time, died, risk = toy()
    assert ((time < 5) & (died == 0)).sum() == 0  # censoring only at 8 years
    expected = np.mean((((time < 5) & (died == 1)).astype(float) - risk) ** 2)
    assert brier5(time, died, risk) == pytest.approx(expected)


def test_brier5_reweights_censored():
    # Two people; the one censored at 2 years drops out, the survivor carries weight 1 / G(5-) = 2.
    time, died, risk = np.array([2.0, 7.0]), np.array([0, 0]), np.array([0.5, 0.2])
    assert brier5(time, died, risk) == pytest.approx((0 + 2 * 0.2**2) / 2)


def test_citl_zero_when_calibrated():
    time, died, _ = toy()
    risk = np.full(len(time), km_risk(time, died, 5))
    assert citl5(time, died, risk) == pytest.approx(0.0, abs=1e-12)
    assert citl5(time, died, risk + 0.05) == pytest.approx(0.05)  # positive = over-predicts


def test_reliability_table():
    time, died, risk = toy()
    table = reliability_table(time, died, risk)
    assert list(table.columns) == ["bin", "n", "mean_pred", "observed"]
    assert table["n"].sum() == len(time) and len(table) == 10
    assert table["mean_pred"].is_monotonic_increasing


def test_bootstrap_contains_point():
    time, died, risk = toy()
    point, lo, hi = bootstrap_ci(c_index, time, died, risk, n_boot=200)
    assert lo <= point <= hi and hi - lo < 0.15


def test_bootstrap_with_nan_draws():
    time = np.arange(1.0, 31.0) / 4
    died = np.zeros(30, dtype=int)
    died[[3, 20]] = 1
    risk = np.linspace(0.9, 0.1, 30)
    point, lo, hi = bootstrap_ci(c_index, time, died, risk, n_boot=300)
    assert np.isfinite([point, lo, hi]).all()


def test_model_diff_identical_is_zero():
    time, died, risk = toy()
    assert model_diff_ci(c_index, time, died, risk, risk, n_boot=100) == (0.0, 0.0, 0.0)

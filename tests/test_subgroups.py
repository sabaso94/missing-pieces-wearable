import pandas as pd

from nhanes_reliable.subgroups import age_band


def test_age_band_edges():
    bands = age_band(pd.Series([20.0, 49.0, 50.0, 64.0, 65.0, 80.0]))
    assert bands.astype(str).tolist() == ["20-49", "20-49", "50-64", "50-64", "65+", "65+"]
    assert list(bands.cat.categories) == ["20-49", "50-64", "65+"] and bands.cat.ordered


import numpy as np

from nhanes_reliable.metrics import c_index, citl5
from nhanes_reliable.subgroups import gap_change_ci, gap_ci, subgroup_table


def toy(n=600, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    t = rng.exponential(1 / (0.08 * np.exp(x)))
    time, died = np.minimum(t, 8.0), (t < 8.0).astype(int)
    return time, died, 1 - np.exp(-0.08 * np.exp(x) * 5), rng.choice(["a", "b"], n)


def test_gap_identical_groups_zero():
    time, died, risk, g = toy()
    assert gap_ci(c_index, time, died, risk, g == "a", g == "a", n_boot=50) == (0.0, 0.0, 0.0)


def test_gap_change_same_model_zero():
    time, died, risk, g = toy()
    assert gap_change_ci(citl5, time, died, risk, risk, g == "a", g == "b", n_boot=50) == (0.0, 0.0, 0.0)


def test_gap_sign():
    time, died, risk, g = toy()
    shifted = np.where(g == "a", np.clip(risk + 0.1, 0, 1), risk)  # over-predict group a only
    gap, lo, hi = gap_ci(citl5, time, died, shifted, g == "a", g == "b", n_boot=200)
    assert lo > 0 and gap > 0.05


def test_subgroup_table_columns():
    time, died, risk, g = toy()
    table = subgroup_table(time, died, risk, pd.Series(g), n_boot=50)
    assert list(table["group"]) == ["a", "b"] and table["n"].sum() == len(time)
    assert {"deaths", "observed5", "mean_pred", "c_index", "c_index_lo", "citl5", "citl5_hi"} <= set(table.columns)


def test_subgroup_no_deaths_nan_not_crash():
    time, died, risk, g = toy()
    died = np.where(g == "b", 0, died)
    table = subgroup_table(time, died, risk, pd.Series(g), n_boot=50)
    assert np.isnan(table.set_index("group").loc["b", "c_index"])
    gap, lo, hi = gap_ci(c_index, time, died, risk, g == "a", g == "b", n_boot=50)
    assert np.isnan(gap)

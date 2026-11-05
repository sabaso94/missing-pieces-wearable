import numpy as np
import pandas as pd
import pytest

from nhanes_reliable.data import BLOCKS, REMOVABLE
from nhanes_reliable.tabular import (
    ACTIVITY_COLUMNS,
    block_flags,
    fit_tab_stats,
    make_tab_features,
    remove_blocks,
    tab_variables,
)


def frame(n=50, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({v: rng.normal(size=n) for b in BLOCKS.values() for v in b})
    for c in ACTIVITY_COLUMNS:
        df[c] = rng.normal(size=n)
    for b in REMOVABLE:
        df[f"has_{b}"] = True
    df.loc[0, "hba1c"] = np.nan
    df.loc[1, "has_labs"] = False
    return df


def test_tab_variables_order():
    assert tab_variables(["exam", "core"]) == BLOCKS["core"] + BLOCKS["exam"]
    assert tab_variables(["core", "wearable"]) == BLOCKS["core"]
    assert tab_variables(["core", "wearable"], with_activity=True) == BLOCKS["core"] + ACTIVITY_COLUMNS


def test_stats_fit_part_only():
    df = frame()
    variables = tab_variables(["core", "labs"])
    fit = np.arange(30)
    stats = fit_tab_stats(df.iloc[fit], variables)
    changed = df.copy()
    changed.loc[30:, variables] = 1e6
    assert fit_tab_stats(changed.iloc[fit], variables) == stats


def test_zero_std_becomes_one():
    df = frame()
    df["age"] = 50.0
    assert fit_tab_stats(df, ["age"])["age"] == (50.0, 1.0)


def test_features_never_nan():
    df = frame()
    variables = tab_variables(["core", "labs"])
    x, names = make_tab_features(df, fit_tab_stats(df, variables), variables)
    assert x.dtype == np.float32 and x.shape == (len(df), 2 * len(variables)) and not np.isnan(x).any()
    assert names[: len(variables)] == variables and names[len(variables)] == f"{variables[0]}_missing"


def test_indicator_matches_missing():
    df = frame()
    variables = ["hba1c"]
    x, _ = make_tab_features(df, fit_tab_stats(df, variables), variables)
    assert x[0, 0] == 0 and x[0, 1] == 1 and (x[1:, 1] == 0).all()


def test_remove_blocks():
    df = frame()
    removed = remove_blocks(df, ["labs", "wearable"])
    assert removed[BLOCKS["labs"]].isna().all().all() and removed[ACTIVITY_COLUMNS].isna().all().all()
    assert removed[BLOCKS["exam"]].notna().all().all()
    assert df[BLOCKS["exam"]].notna().all().all() and df["albumin"].notna().all()  # original untouched


def test_remove_core_raises():
    with pytest.raises(ValueError):
        remove_blocks(frame(), ["core"])


def test_remove_unknown_raises():
    with pytest.raises(ValueError):
        remove_blocks(frame(), ["imaging"])


def test_block_flags_removed_forced_off():
    df = frame()
    flags = block_flags(df)
    assert flags.shape == (len(df), 4) and flags.dtype == np.float32
    assert flags[1, REMOVABLE.index("labs")] == 0 and flags[0].sum() == 4
    flags = block_flags(df, removed=["wearable"])
    assert (flags[:, 0] == 0).all() and (flags[:, 2] == 1).all()

import numpy as np
import pandas as pd

from nhanes_reliable.folds import assign_folds, inner_split


def cohort(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"died": (rng.random(n) < 0.1).astype(int),
                         "cycle": rng.choice(["2011-12", "2013-14"], n)},
                        index=pd.Index(np.arange(1000, 1000 + n), name="seqn"))


def test_folds_stratified():
    c = cohort()
    folds = assign_folds(c)
    assert folds.index.equals(c.index) and set(folds) == {0, 1, 2, 3, 4}
    overall = c["died"].mean()
    for k in range(5):
        assert abs(c["died"][folds == k].mean() - overall) < 0.015
    assert assign_folds(c).equals(folds)


def test_inner_split_disjoint():
    c = cohort()
    folds = assign_folds(c).to_numpy()
    train = np.flatnonzero(folds != 2)
    fit, hold = inner_split(train, c["died"].to_numpy(), fold=2)
    assert len(np.intersect1d(fit, hold)) == 0
    np.testing.assert_array_equal(np.sort(np.r_[fit, hold]), train)
    assert abs(len(hold) / len(train) - 0.2) < 0.01
    np.testing.assert_array_equal(inner_split(train, c["died"].to_numpy(), fold=2)[1], hold)

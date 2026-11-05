import numpy as np
import pandas as pd

from nhanes_reliable import crossfit as cf
from nhanes_reliable.data import BLOCKS, REMOVABLE
from nhanes_reliable.folds import assign_folds
from nhanes_reliable.tabular import ACTIVITY_COLUMNS


def synthetic(n=400, seed=0):
    rng = np.random.default_rng(seed)
    cohort = pd.DataFrame({v: rng.normal(size=n) for b in BLOCKS.values() for v in b},
                          index=pd.Index(np.arange(n) + 100, name="seqn"))
    for c in ACTIVITY_COLUMNS:
        cohort[c] = rng.normal(size=n)
    t = rng.exponential(1 / (0.08 * np.exp(cohort["age"].to_numpy())))
    cohort["time_years"], cohort["died"] = np.minimum(t, 9.5), (t < 9.5).astype(int)
    cohort["cycle"] = rng.choice(["2011-12", "2013-14"], n)
    cohort.loc[cohort.index[:40], BLOCKS["labs"]] = np.nan
    for b in REMOVABLE:
        cohort[f"has_{b}"] = True
    cohort["has_labs"] = cohort[BLOCKS["labs"]].notna().any(axis=1)
    cohort["complete_case"] = cohort[[f"has_{b}" for b in REMOVABLE]].all(axis=1)
    return cohort


def test_crossfit_every_person_one_oof(tmp_path):
    cohort = synthetic()
    folds = assign_folds(cohort)
    conditions = {"none": [], "no_labs": ["labs"]}
    preds = cf.crossfit("cox_clinical", cohort, None, folds, conditions, device="cpu", preds_dir=tmp_path)
    assert set(preds.columns) == {"seqn", "fold", "split", "condition", "risk5"}
    oof = preds[preds["split"] == "oof"]
    for condition in conditions:
        rows = oof[oof["condition"] == condition]
        assert rows["seqn"].is_unique and set(rows["seqn"]) == set(cohort.index)
        assert (rows["fold"].to_numpy() == folds.loc[rows["seqn"]].to_numpy()).all()
    hold = preds[preds["split"] == "hold"]
    assert (hold["fold"].to_numpy() != folds.loc[hold["seqn"]].to_numpy()).all()  # never the person's test fold
    assert (tmp_path / "cox_clinical.csv").exists()


def test_train_on_complete_excludes_incomplete(tmp_path, monkeypatch):
    cohort = synthetic()
    seen = []
    original = cf.fit_cox

    def spy(X, time, died, **kwargs):
        seen.append(X.index)
        return original(X, time, died, **kwargs)

    monkeypatch.setattr(cf, "fit_cox", spy)
    cf.crossfit("cox_subset", cohort, None, assign_folds(cohort), {"none": [], "no_labs": ["labs"]},
                device="cpu", train_on="complete", preds_dir=tmp_path)
    assert seen and all(cohort.loc[index, "complete_case"].all() for index in seen)


def test_cox_subset_uses_only_present_blocks(tmp_path):
    cohort = synthetic()
    folds = assign_folds(cohort)
    a = cf.crossfit("cox_subset", cohort, None, folds, {"no_labs": ["labs"]}, device="cpu", preds_dir=tmp_path)
    changed = cohort.copy()
    changed[BLOCKS["labs"]] = 123.0  # labs are not an input once removed, so predictions cannot change
    b = cf.crossfit("cox_subset", changed, None, folds, {"no_labs": ["labs"]}, device="cpu", preds_dir=tmp_path)
    np.testing.assert_allclose(a["risk5"], b["risk5"])

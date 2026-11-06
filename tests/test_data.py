from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nhanes_reliable.data import (
    BLOCKS,
    REMOVABLE,
    build_cohort,
    clean_clinical,
    load_clinical,
    load_mortality,
    read_mortality_file,
)

FIXTURES = Path(__file__).parent / "fixtures"
DATA = Path(__file__).resolve().parents[1] / "data"


def raw_row(**overrides) -> dict:
    """One NHANES respondent with plausible raw values; overrides replace single fields."""
    row = {
        "SEQN": 1, "RIDAGEYR": 50, "RIAGENDR": 2, "RIDRETH3": 3, "INDFMPIR": 2.5, "DMDEDUC2": 4,
        "BMXBMI": 27.0, "BMXWAIST": 95.0, "BPXSY1": 120, "BPXSY2": 124, "BPXSY3": np.nan, "BPXSY4": np.nan,
        "BPXDI1": 80, "BPXDI2": 0, "BPXDI3": np.nan, "BPXDI4": np.nan, "MGDCGSZ": 60.0,
        "LBXGH": 5.6, "LBXTC": 190, "LBDHDD": 50, "LBXSCR": 0.9, "LBXSAL": 4.2, "LBXSAPSI": 70,
        "LBXWBCSI": 6.5, "LBXHGB": 14.0, "LBXRDW": 13.0,
        "SMQ020": 1, "SMQ040": 3, "HUQ010": 2, "DIQ010": 2,
        "MCQ160B": 2, "MCQ160C": 2, "MCQ160E": 2, "MCQ160F": 2, "MCQ220": 2,
    }
    row.update(overrides)
    return row


def test_special_codes_to_nan():
    raw = pd.DataFrame([raw_row(SEQN=1, SMQ020=7), raw_row(SEQN=2, SMQ020=9, DMDEDUC2=7),
                        raw_row(SEQN=3, DMDEDUC2=9, HUQ010=9), raw_row(SEQN=4, MCQ220=9, DIQ010=7)])
    clean = clean_clinical(raw, cycle="2011-12")
    assert clean.loc[1, ["current_smoker", "former_smoker"]].isna().all()
    assert clean.loc[2, ["current_smoker", "former_smoker", "education"]].isna().all()
    assert np.isnan(clean.loc[3, "education"]) and np.isnan(clean.loc[3, "self_rated_health"])
    assert np.isnan(clean.loc[4, "cancer"]) and np.isnan(clean.loc[4, "diabetes"])


def test_derived_variables():
    raw = pd.DataFrame([raw_row(SEQN=1), raw_row(SEQN=2, SMQ020=2, SMQ040=np.nan, DIQ010=3),
                        raw_row(SEQN=3, SMQ040=2, RIAGENDR=1)])
    clean = clean_clinical(raw, cycle="2013-14")
    assert clean.loc[1, "former_smoker"] == 1 and clean.loc[1, "current_smoker"] == 0
    assert clean.loc[2, "former_smoker"] == 0 and clean.loc[2, "current_smoker"] == 0  # never smoked
    assert clean.loc[2, "diabetes"] == 0  # borderline counts as no diagnosis
    assert clean.loc[3, "current_smoker"] == 1 and clean.loc[3, "female"] == 0
    assert clean.loc[1, "sbp"] == 122  # mean of the available readings
    assert clean.loc[1, "dbp"] == 80  # a diastolic 0 means "not detectable", not a value
    assert clean.loc[1, "race"] == "NH White" and clean.loc[1, "cycle"] == "2013-14"


def test_mortality_fixed_width():
    mort = read_mortality_file(FIXTURES / "mort_sample.dat")
    assert mort.loc[62161, "eligstat"] == 1 and mort.loc[62161, "died"] == 0 and mort.loc[62161, "months"] == 92
    assert np.isnan(mort.loc[62162, "months"]) and mort.loc[62162, "eligstat"] == 2
    assert mort.loc[62174, "died"] == 1 and mort.loc[62174, "months"] == 21


def cohort_inputs():
    clinical = pd.DataFrame(
        {"age": [18, 20, 60, 45], "female": [1, 0, 1, 0], "race": ["NH White"] * 4, "cycle": ["2011-12"] * 4},
        index=pd.Index([1, 2, 3, 4], name="seqn"),
    )
    for block in ["labs", "exam", "questionnaire"]:
        for var in BLOCKS[block]:
            clinical[var] = 1.0
    mortality = pd.DataFrame({"eligstat": [1, 1, 1, 2], "died": [0, 1, 0, 0], "months": [90.0, 30.0, 70.0, np.nan]},
                             index=pd.Index([1, 2, 3, 4], name="seqn"))
    n_valid = pd.Series([7, 3], index=pd.Index([2, 3], name="seqn"))
    return clinical, mortality, n_valid


def test_cohort_filters():
    cohort = build_cohort(*cohort_inputs())
    assert list(cohort.index) == [2, 3]  # 18-year-old and ineligible person dropped
    assert cohort.loc[2, "time_years"] == pytest.approx(2.5)


def test_block_flags():
    clinical, mortality, n_valid = cohort_inputs()
    clinical.loc[2, BLOCKS["labs"]] = np.nan  # no lab value at all
    clinical.loc[3, BLOCKS["labs"][0]] = np.nan  # one lab value missing
    cohort = build_cohort(clinical, mortality, n_valid)
    assert not cohort.loc[2, "has_labs"] and cohort.loc[3, "has_labs"]
    assert cohort.loc[2, "has_wearable"] and cohort.loc[3, "n_valid_days"] == 3
    expected = cohort[[f"has_{b}" for b in REMOVABLE]].all(axis=1)
    assert (cohort["complete_case"] == expected).all()
    assert not cohort.loc[2, "complete_case"] and cohort.loc[3, "complete_case"]


def test_missing_wear_counts_as_zero_days():
    clinical, mortality, _ = cohort_inputs()
    cohort = build_cohort(clinical, mortality, pd.Series([], dtype=int, index=pd.Index([], name="seqn")))
    assert (cohort["n_valid_days"] == 0).all() and not cohort["has_wearable"].any()


def test_race_not_in_blocks():
    assert all("race" not in variables for variables in BLOCKS.values())
    assert BLOCKS["core"] == ["age", "female"] and BLOCKS["wearable"] == []
    assert list(BLOCKS) == ["core", "wearable", "labs", "exam", "questionnaire"]


@pytest.mark.skipif(not (DATA / "mortality").exists() or not (DATA / "nhanes").exists(), reason="data not downloaded")
def test_real_cohort():
    clinical, mortality = load_clinical(DATA), load_mortality(DATA)
    cohort = build_cohort(clinical, mortality, pd.Series([], dtype=int, index=pd.Index([], name="seqn")))
    assert 9_000 <= len(cohort) <= 12_500
    assert 700 <= cohort["died"].sum() <= 1_400
    assert cohort["time_years"].max() < 10
    assert cohort.index.is_unique


def test_xpt_zero_diastolic_is_nan():
    """pandas reads an XPT zero as 5.4e-79; a diastolic reading of 0 ("not detectable") must still become NaN."""
    raw = pd.DataFrame([raw_row(SEQN=1, BPXDI2=5.397605346934028e-79)])
    clean = clean_clinical(raw, cycle="2011-12")
    assert clean.loc[1, "dbp"] == 80

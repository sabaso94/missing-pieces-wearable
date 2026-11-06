"""NHANES 2011-2014 clinical variables, linked mortality, and the analysis cohort.

Every model input belongs to one block. The blocks are what gets removed in the missing-data
experiments: core (age, sex) is always present; wearable, labs, exam and questionnaire can be missing.
Race/ethnicity is not a model input; it is only used to compare groups.
"""

from pathlib import Path

import numpy as np
import pandas as pd

BLOCKS: dict[str, list[str]] = {
    "core": ["age", "female"],
    "wearable": [],  # the minute-level days (deep models) or the activity summaries (Cox), see activity.py
    "labs": ["hba1c", "total_chol", "hdl", "creatinine", "albumin", "alk_phos", "wbc", "hemoglobin", "rdw"],
    "exam": ["bmi", "waist", "sbp", "dbp", "grip"],
    "questionnaire": [
        "current_smoker", "former_smoker", "self_rated_health", "education", "income_poverty",
        "diabetes", "heart_failure", "coronary_disease", "heart_attack", "stroke", "cancer",
    ],
}
REMOVABLE: list[str] = ["wearable", "labs", "exam", "questionnaire"]

# RIDRETH3: 1 Mexican American, 2 other Hispanic, 3 non-Hispanic White, 4 non-Hispanic Black,
# 6 non-Hispanic Asian, 7 other or multiracial.
RACE_GROUPS: dict[int, str] = {1: "Hispanic", 2: "Hispanic", 3: "NH White", 4: "NH Black", 6: "NH Asian/other",
                               7: "NH Asian/other"}

CYCLES: dict[str, str] = {"G": "2011-12", "H": "2013-14"}
FILES = ["DEMO", "BMX", "BPX", "MGX", "GHB", "TCHOL", "HDL", "BIOPRO", "CBC", "SMQ", "HUQ", "MCQ", "DIQ"]
MORTALITY_FILES = {"2011-12": "NHANES_2011_2012_MORT_2019_PUBLIC.dat", "2013-14": "NHANES_2013_2014_MORT_2019_PUBLIC.dat"}

# Lab and exam variables copied as they are (continuous, no special codes).
DIRECT = {
    "hba1c": "LBXGH", "total_chol": "LBXTC", "hdl": "LBDHDD", "creatinine": "LBXSCR", "albumin": "LBXSAL",
    "alk_phos": "LBXSAPSI", "wbc": "LBXWBCSI", "hemoglobin": "LBXHGB", "rdw": "LBXRDW",
    "bmi": "BMXBMI", "waist": "BMXWAIST", "grip": "MGDCGSZ", "income_poverty": "INDFMPIR",
}
# Yes/no questions (1 = yes, 2 = no; 7 = refused and 9 = don't know become missing).
YES_NO = {"heart_failure": "MCQ160B", "coronary_disease": "MCQ160C", "heart_attack": "MCQ160E", "stroke": "MCQ160F",
          "cancer": "MCQ220"}


def _valid_codes(values: pd.Series, allowed: list[int]) -> pd.Series:
    """Keep only the listed answer codes; refused (7, 77, ...), don't know (9, 99, ...) and blanks become NaN."""
    return values.where(values.isin(allowed))


def clean_clinical(raw: pd.DataFrame, cycle: str) -> pd.DataFrame:
    """Turn raw NHANES columns (one row per SEQN) into the model variables of BLOCKS plus race and cycle."""
    out = pd.DataFrame(index=pd.Index(raw["SEQN"].astype(int), name="seqn"))
    raw = raw.set_index(out.index)
    # pandas reads an XPT zero as 5.4e-79; make it an exact 0 so zero codes are recognised.
    numeric = raw.select_dtypes("number").columns
    raw[numeric] = raw[numeric].mask(raw[numeric].abs() < 1e-70, 0.0)

    out["age"] = raw["RIDAGEYR"].astype(float)  # NHANES top-codes age at 80
    out["age_raw"] = out["age"]
    out["female"] = (raw["RIAGENDR"] == 2).astype(float)
    out["race"] = raw["RIDRETH3"].map(RACE_GROUPS)
    out["cycle"] = cycle

    for name, column in DIRECT.items():
        out[name] = raw[column].astype(float)
    # Blood pressure: mean of up to four readings. A diastolic reading of 0 means "not detectable".
    out["sbp"] = raw[[f"BPXSY{i}" for i in range(1, 5)]].mean(axis=1)
    out["dbp"] = raw[[f"BPXDI{i}" for i in range(1, 5)]].replace(0, np.nan).mean(axis=1)

    # Smoking: SMQ020 = smoked at least 100 cigarettes in life; SMQ040 = smokes now (1 every day, 2 some days,
    # 3 not at all). Never smokers are the reference (both indicators 0).
    ever = _valid_codes(raw["SMQ020"], [1, 2])
    now = _valid_codes(raw["SMQ040"], [1, 2, 3])
    out["current_smoker"] = np.where(ever == 2, 0.0, np.where(ever == 1, now.isin([1, 2]).astype(float), np.nan))
    out["former_smoker"] = np.where(ever == 2, 0.0, np.where(ever == 1, (now == 3).astype(float), np.nan))
    # An ever smoker with no answer on current smoking is unknown, not "former".
    unknown_now = (ever == 1) & now.isna()
    out.loc[unknown_now, ["current_smoker", "former_smoker"]] = np.nan

    out["self_rated_health"] = _valid_codes(raw["HUQ010"], [1, 2, 3, 4, 5]).astype(float)  # 1 excellent .. 5 poor
    out["education"] = _valid_codes(raw["DMDEDUC2"], [1, 2, 3, 4, 5]).astype(float)  # 1 < 9th grade .. 5 college
    # DIQ010: 1 yes, 2 no, 3 borderline. Borderline is not a diagnosis of diabetes.
    diabetes = _valid_codes(raw["DIQ010"], [1, 2, 3])
    out["diabetes"] = np.where(diabetes.isna(), np.nan, (diabetes == 1).astype(float))
    for name, column in YES_NO.items():
        answer = _valid_codes(raw[column], [1, 2])
        out[name] = np.where(answer.isna(), np.nan, (answer == 1).astype(float))
    return out


def load_clinical(data_dir: Path) -> pd.DataFrame:
    """Read the NHANES XPT files of both cycles and return the cleaned variables, one row per SEQN."""
    frames = []
    for suffix, cycle in CYCLES.items():
        raw = None
        for name in FILES:
            table = pd.read_sas(data_dir / "nhanes" / f"{name}_{suffix}.xpt", format="xport")
            raw = table if raw is None else raw.merge(table, on="SEQN", how="left")  # DEMO first: everyone
        frames.append(clean_clinical(raw, cycle))
    return pd.concat(frames)


def read_mortality_file(path: Path) -> pd.DataFrame:
    """One NCHS public-use linked mortality file (fixed width; positions from the NCHS read-in program)."""
    table = pd.read_fwf(path, colspecs=[(0, 6), (14, 15), (15, 16), (45, 48)],
                        names=["seqn", "eligstat", "mortstat", "months"], na_values=["."], header=None)
    table = table.set_index(table["seqn"].astype(int).rename("seqn"))
    # ELIGSTAT: 1 eligible, 2 under 18 (not released), 3 ineligible. MORTSTAT: 0 alive, 1 dead.
    return pd.DataFrame({"eligstat": table["eligstat"], "died": table["mortstat"], "months": table["months"]})


def load_mortality(data_dir: Path) -> pd.DataFrame:
    """Linked mortality for both cycles; months = PERMTH_EXM, months from the exam to death or end of 2019."""
    return pd.concat([read_mortality_file(data_dir / "mortality" / f) for f in MORTALITY_FILES.values()])


def build_cohort(clinical: pd.DataFrame, mortality: pd.DataFrame, n_valid_days: pd.Series) -> pd.DataFrame:
    """Adults aged 20+ who are eligible for mortality follow-up and were examined, with block-present flags.

    A block counts as present when at least one of its variables is non-missing; single missing values inside
    a present block are imputed later. The wearable block is present with at least one valid wear day.
    """
    cohort = clinical.join(mortality, how="inner")
    cohort = cohort[(cohort["age"] >= 20) & (cohort["eligstat"] == 1) & cohort["months"].notna()].copy()
    cohort["died"] = cohort["died"].astype(int)
    cohort["time_years"] = cohort["months"] / 12
    cohort["n_valid_days"] = n_valid_days.reindex(cohort.index).fillna(0).astype(int)
    cohort["has_wearable"] = cohort["n_valid_days"] >= 1
    for block in ["labs", "exam", "questionnaire"]:
        cohort[f"has_{block}"] = cohort[BLOCKS[block]].notna().any(axis=1)
    cohort["complete_case"] = cohort[[f"has_{b}" for b in REMOVABLE]].all(axis=1)
    return cohort

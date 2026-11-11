"""E6: do more active people die less, or do people who are already ill move less?

Among complete cases with a known 5-year status, the most active third (daily active minutes) is compared with
the least active third (the middle third is left out). Risk of death within 5 years if everyone were in the top
third against everyone in the bottom third, with four adjustment sets: none; age and sex; factors that come before
the activity level (PRE_EXISTING); and every clinical input of the models, which adds factors that activity
itself may change (BMI, waist, blood pressure, labs, grip strength, diabetes) and two checks for reverse causation: leaving out deaths in the first 2 years, and keeping only
people healthy at baseline (self-rated health excellent to good, none of heart failure, coronary disease, heart
attack, stroke or cancer). Survey weights are not used: the estimates describe this sample.
Writes results/e6_activity.csv and results/e6_overlap.csv.
"""
import os

for _var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(_var, "1")  # small models: one thread is fastest and keeps results identical

from pathlib import Path

import numpy as np
import pandas as pd

from nhanes_reliable.causal import aipw_risk, e_value, e_value_ci, overlap, risk_unadjusted
from nhanes_reliable.crossfit import prepare
from nhanes_reliable.tabular import fit_tab_stats, make_tab_features, tab_variables

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CONDITIONS = ["heart_failure", "coronary_disease", "heart_attack", "stroke", "cancer"]
# Factors that precede a person's activity level rather than follow from it: confounders, not pathways.
PRE_EXISTING = ["age", "female", "current_smoker", "former_smoker", "education", "income_poverty",
                "self_rated_health", *CONDITIONS]
NUISANCE_LABEL = {"linear": "logistic", "boosting": "boosted"}


def analysis_sets(cohort: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Each analysis population with columns exposed (1 top third, 0 bottom third) and died5."""
    base = cohort[cohort["complete_case"]]
    base = base[(base["time_years"] >= 5) | (base["died"] == 1)].copy()
    base["died5"] = ((base["died"] == 1) & (base["time_years"] < 5)).astype(int)
    low, high = base["act_active_minutes"].quantile([1 / 3, 2 / 3])
    base = base[(base["act_active_minutes"] < low) | (base["act_active_minutes"] >= high)].copy()
    base["exposed"] = (base["act_active_minutes"] >= high).astype(int)
    early = (base["died"] == 1) & (base["time_years"] < 2)
    healthy = (base["self_rated_health"] <= 3) & ~(base[CONDITIONS] == 1).any(axis=1)
    return {"all": base, "deaths in the first 2 years left out": base[~early],
            "healthy at baseline": base[healthy]}


def adjustment_sets(df: pd.DataFrame) -> dict[str, np.ndarray]:
    def features(variables):
        x, _ = make_tab_features(df, fit_tab_stats(df, variables), variables)
        return x.astype(float)

    return {"age and sex": df[["age", "female"]].to_numpy(float),
            "pre-existing factors": features(PRE_EXISTING),
            "all clinical inputs": features(tab_variables(["core", "labs", "exam", "questionnaire"]))}


def main() -> None:
    cohort, _ = prepare(ROOT / "data")
    rows, overlaps = [], []
    for analysis, df in analysis_sets(cohort).items():
        y, t = df["died5"].to_numpy(), df["exposed"].to_numpy()
        info = {"analysis": analysis, "n": len(df), "n_top_third": int(t.sum()), "deaths": int(y.sum())}
        rows.append(info | {"adjustment": "none", "nuisance": "-"} | risk_unadjusted(y, t))
        for adjustment, x in adjustment_sets(df).items():
            for nuisance in (["linear"] if adjustment == "age and sex" else ["linear", "boosting"]):
                out, e = aipw_risk(y, t, x, nuisance, return_propensity=True)
                label = NUISANCE_LABEL[nuisance]
                row = info | {"adjustment": adjustment, "nuisance": label} | out
                row["e_value"] = e_value(out["rr"])
                row["e_value_ci"] = e_value_ci(out["rr"], out["rr_low"], out["rr_high"])
                rows.append(row)
                overlaps.append({"analysis": analysis, "adjustment": adjustment, "nuisance": label} | overlap(e))
    results = pd.DataFrame(rows)
    results.to_csv(RESULTS / "e6_activity.csv", index=False, float_format="%.4f")
    pd.DataFrame(overlaps).to_csv(RESULTS / "e6_overlap.csv", index=False, float_format="%.4f")
    print(results.drop(columns=["rd_low", "rd_high"]).round(3).to_string(index=False))
    print(pd.DataFrame(overlaps).round(3).to_string(index=False))


if __name__ == "__main__":
    main()

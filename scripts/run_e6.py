"""E6: do more active people die less, or do people who are already ill move less?

Among complete cases with a known 5-year status, the most active third (daily active minutes) is compared with
the least active third (the middle third is left out). Risk of death within 5 years if everyone were in the top
third against everyone in the bottom third, with three adjustment sets (none; age and sex; every clinical input of
the models) and two checks for reverse causation: leaving out deaths in the first 2 years, and keeping only
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

from nhanes_reliable.causal import aipw_risk, e_value, overlap, risk_unadjusted
from nhanes_reliable.crossfit import prepare
from nhanes_reliable.tabular import fit_tab_stats, make_tab_features, tab_variables

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CONDITIONS = ["heart_failure", "coronary_disease", "heart_attack", "stroke", "cancer"]


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
    variables = tab_variables(["core", "labs", "exam", "questionnaire"])
    full, _ = make_tab_features(df, fit_tab_stats(df, variables), variables)
    return {"age and sex": df[["age", "female"]].to_numpy(float), "all clinical inputs": full.astype(float)}


def main() -> None:
    cohort, _ = prepare(ROOT / "data")
    rows, overlaps = [], []
    for analysis, df in analysis_sets(cohort).items():
        y, t = df["died5"].to_numpy(), df["exposed"].to_numpy()
        info = {"analysis": analysis, "n": len(df), "n_top_third": int(t.sum()), "deaths": int(y.sum())}
        rows.append(info | {"adjustment": "none", "nuisance": "-"} | risk_unadjusted(y, t))
        for adjustment, x in adjustment_sets(df).items():
            for nuisance in (["linear", "boosting"] if adjustment == "all clinical inputs" else ["linear"]):
                out, e = aipw_risk(y, t, x, nuisance, return_propensity=True)
                row = info | {"adjustment": adjustment, "nuisance": nuisance} | out
                row["e_value"], row["e_value_ci"] = e_value(out["rr"]), e_value(min(out["rr_high"], 1.0))
                rows.append(row)
                overlaps.append({"analysis": analysis, "adjustment": adjustment, "nuisance": nuisance} | overlap(e))
    results = pd.DataFrame(rows)
    results.to_csv(RESULTS / "e6_activity.csv", index=False, float_format="%.4f")
    pd.DataFrame(overlaps).to_csv(RESULTS / "e6_overlap.csv", index=False, float_format="%.4f")
    print(results.drop(columns=["rd_low", "rd_high"]).round(3).to_string(index=False))
    print(pd.DataFrame(overlaps).round(3).to_string(index=False))


if __name__ == "__main__":
    main()

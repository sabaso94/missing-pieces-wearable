"""Cohort report: who is in the cohort, who died, how long they were followed, what data they have.

Also fixes the subgroups used in E4-E5: any age band or race group with fewer than MIN_DEATHS deaths in the
complete-case set is merged with a neighbour. Writes results/cohort.csv, results/missing_blocks.csv,
results/groups.json and results/folds.csv.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from nhanes_reliable.crossfit import prepare
from nhanes_reliable.data import REMOVABLE
from nhanes_reliable.folds import assign_folds
from nhanes_reliable.subgroups import AGE_EDGES, age_band

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MIN_DEATHS = 40


def summarise(df: pd.DataFrame, name: str) -> dict:
    return {"set": name, "n": len(df), "deaths": int(df["died"].sum()),
            "deaths_5y": int(((df["died"] == 1) & (df["time_years"] < 5)).sum()),
            "median_followup_years": round(float(df["time_years"].median()), 2),
            "censored_before_5y": int(((df["died"] == 0) & (df["time_years"] < 5)).sum()),
            "mean_age": round(float(df["age"].mean()), 1), "female": round(float(df["female"].mean()), 3)}


def merge_small(groups: pd.Series, died: pd.Series, order: list[str]) -> dict[str, str]:
    """Map each group to itself, or merge one with fewer than MIN_DEATHS deaths into its smaller neighbour."""
    mapping = {g: g for g in order}
    while True:
        labels = groups.map(mapping)
        deaths = died.groupby(labels).sum().reindex(dict.fromkeys(mapping.values()).keys())
        small = deaths[deaths < MIN_DEATHS]
        if small.empty or len(deaths) == 1:
            return mapping
        current = list(deaths.index)
        g = small.index[0]
        i = current.index(g)
        neighbours = [current[j] for j in (i - 1, i + 1) if 0 <= j < len(current)]
        target = min(neighbours, key=lambda h: deaths[h])
        merged = f"{target}+{g}" if current.index(target) < i else f"{g}+{target}"
        mapping = {k: (merged if v in (g, target) else v) for k, v in mapping.items()}


def main() -> None:
    cohort, _ = prepare(ROOT / "data")
    RESULTS.mkdir(exist_ok=True)
    complete = cohort[cohort["complete_case"]]

    table = pd.DataFrame([summarise(cohort, "cohort"), summarise(complete, "complete_case")])
    table.to_csv(RESULTS / "cohort.csv", index=False)
    print(table.to_string(index=False))

    valid_days = cohort["n_valid_days"].value_counts().sort_index()
    print("\nvalid wear days in the cohort:", valid_days.to_dict())

    missing = pd.DataFrame({f"no_{b}": ~cohort[f"has_{b}"] for b in REMOVABLE})
    missing_rows = []
    for column in missing:
        people = cohort[missing[column]]
        missing_rows.append({"missing": column, "n": len(people), "deaths": int(people["died"].sum()),
                             "mean_age": round(float(people["age"].mean()), 1) if len(people) else np.nan})
    missing_table = pd.DataFrame(missing_rows)
    missing_table.to_csv(RESULTS / "missing_blocks.csv", index=False)
    print("\n" + missing_table.to_string(index=False))

    age = age_band(complete["age"]).astype(str)
    age_order = [str(c) for c in age_band(pd.Series([float(e) for e in AGE_EDGES])).cat.categories]
    race_order = ["NH White", "NH Black", "Hispanic", "NH Asian/other"]
    groups = {"age_band": merge_small(age, complete["died"], age_order),
              "race": merge_small(complete["race"], complete["died"], race_order),
              "sex": {"male": "male", "female": "female"}}
    print("\ndeaths per group (complete cases):")
    print(" age:", complete["died"].groupby(age).sum().to_dict())
    print(" race:", complete["died"].groupby(complete["race"]).sum().to_dict())
    print(" sex:", complete["died"].groupby(complete["female"]).sum().rename({0: "male", 1: "female"}).to_dict())
    print("final groups:", groups)
    (RESULTS / "groups.json").write_text(json.dumps(groups, indent=2))

    folds = assign_folds(cohort)
    folds.to_csv(RESULTS / "folds.csv")


if __name__ == "__main__":
    main()

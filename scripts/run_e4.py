"""E4: subgroups. Is the model equally good at ranking, and equally well calibrated, for every group?

Groups (fixed by cohort_report.py in results/groups.json): sex, age band, race/ethnicity. Race is never a model
input. Scored on the complete-case out-of-fold predictions of cox_full, fused, and fused with the labs removed.
Gaps are against a reference group (male, 50-64, NH White), with person-level bootstrap CIs.
A C-index inside one age band is lower by construction (age varies less), so C-indices are compared across
models, not across groups; calibration-in-the-large (citl5) is the main fairness quantity.
Writes results/e4_groups.csv and results/e4_gaps.csv.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from nhanes_reliable.crossfit import load_preds, prepare
from nhanes_reliable.crossfit import oof_risk
from nhanes_reliable.metrics import c_index, citl5
from nhanes_reliable.subgroups import age_band, gap_ci, subgroup_table

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MODELS = {"cox_full": ("cox_full", "none"), "fused": ("fused", "none"), "fused_no_labs": ("fused", "no_labs")}
REFERENCES = {"sex": "male", "age_band": "50-64", "race": "NH White"}


def group_labels(cohort: pd.DataFrame) -> pd.DataFrame:
    """sex, age_band and race labels per person, with the merges from results/groups.json applied."""
    mapping = json.loads((RESULTS / "groups.json").read_text())
    labels = pd.DataFrame(index=cohort.index)
    labels["sex"] = cohort["female"].map({0.0: "male", 1.0: "female"})
    labels["age_band"] = age_band(cohort["age"]).astype(str).map(mapping["age_band"])
    labels["race"] = cohort["race"].map(mapping["race"])
    return labels


def reference_of(attribute: str, labels: pd.Series) -> str:
    """The (possibly merged) group that contains the reference group."""
    return next(g for g in labels.unique() if REFERENCES[attribute] in g.split("+"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()

    cohort, _ = prepare(ROOT / "data")
    complete = cohort[cohort["complete_case"]]
    labels = group_labels(complete)
    time, died = complete["time_years"].to_numpy(), complete["died"].to_numpy()

    tables, gaps = [], []
    for name, (model, condition) in MODELS.items():
        risk = oof_risk(load_preds(model), condition, complete.index)
        for attribute in REFERENCES:
            groups = labels[attribute]
            tables.append(subgroup_table(time, died, risk, groups, args.n_boot).assign(model=name, attribute=attribute))
            reference = reference_of(attribute, groups)
            for group in sorted(set(groups) - {reference}):
                for metric_name, metric in [("c_index", c_index), ("citl5", citl5)]:
                    gap, lo, hi = gap_ci(metric, time, died, risk, (groups == group).to_numpy(),
                                         (groups == reference).to_numpy(), args.n_boot)
                    gaps.append({"model": name, "attribute": attribute, "group": group, "versus": reference,
                                 "metric": metric_name, "gap": gap, "lo": lo, "hi": hi})
        print("evaluated", name, flush=True)

    table = pd.concat(tables)
    table = table[["model", "attribute"] + [c for c in table if c not in ("model", "attribute")]]
    gaps = pd.DataFrame(gaps)
    table.to_csv(RESULTS / "e4_groups.csv", index=False)
    gaps.to_csv(RESULTS / "e4_gaps.csv", index=False)
    print(table.round(4).to_string(index=False))
    print(gaps.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

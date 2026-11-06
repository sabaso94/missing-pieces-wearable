"""E5: recalibration. Can the calibration gaps of E4 be closed, and at what cost overall?

For the fused model, per outer fold: fit a logistic recalibration of the 5-year risk on that fold's hold-out
predictions (complete cases), and apply it to the same fold's out-of-fold predictions. Variants:
- fused:                  as it is
- fused_recal_global:     one recalibration for everyone
- fused_recal_<attribute>: one recalibration per group of sex, age band or race (needs the group at use time)
A gap counts as narrowed only if the CI of its change against fused (same resampled people) excludes zero.
Writes results/e5_groups.csv, e5_gaps.csv, e5_overall.csv.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from nhanes_reliable.crossfit import load_preds, prepare
from nhanes_reliable.metrics import bootstrap_ci, brier5, c_index, citl5, model_diff_ci
from nhanes_reliable.recalibration import apply_group_recal, apply_recal, fit_group_recal, fit_recal
from nhanes_reliable.subgroups import gap_change_ci, gap_ci, subgroup_table

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_e4 import REFERENCES, group_labels, reference_of  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
METRICS = {"c_index": c_index, "brier5": brier5, "citl5": citl5}


def recalibrated(cohort: pd.DataFrame, labels: pd.DataFrame, preds: pd.DataFrame) -> dict[str, pd.Series]:
    """Out-of-fold risk of every variant, indexed by seqn (complete cases only)."""
    preds = preds[(preds["condition"] == "none") & preds["seqn"].isin(cohort.index)]
    variants = {name: [] for name in ["fused", "fused_recal_global"] + [f"fused_recal_{a}" for a in REFERENCES]}
    for _, fold in preds.groupby("fold"):
        hold = fold[fold["split"] == "hold"].set_index("seqn")
        oof = fold[fold["split"] == "oof"].set_index("seqn")
        t, d = cohort.loc[hold.index, "time_years"].to_numpy(), cohort.loc[hold.index, "died"].to_numpy()
        r_hold, r_oof = hold["risk5"].to_numpy(), oof["risk5"].to_numpy()
        variants["fused"].append(pd.Series(r_oof, index=oof.index))
        recal = apply_recal(r_oof, fit_recal(t, d, r_hold))
        # Monotone within one fold: the fold's own ranking is unchanged.
        t_oof, d_oof = cohort.loc[oof.index, "time_years"].to_numpy(), cohort.loc[oof.index, "died"].to_numpy()
        if abs(c_index(t_oof, d_oof, recal) - c_index(t_oof, d_oof, r_oof)) > 1e-4:
            raise AssertionError("global recalibration changed the ranking within a fold")
        variants["fused_recal_global"].append(pd.Series(recal, index=oof.index))
        for attribute in REFERENCES:
            params = fit_group_recal(t, d, r_hold, labels.loc[hold.index, attribute].to_numpy())
            out = apply_group_recal(r_oof, labels.loc[oof.index, attribute].to_numpy(), params)
            variants[f"fused_recal_{attribute}"].append(pd.Series(out, index=oof.index))
    return {name: pd.concat(parts).loc[cohort.index] for name, parts in variants.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()

    cohort, _ = prepare(ROOT / "data")
    complete = cohort[cohort["complete_case"]]
    labels = group_labels(complete)
    time, died = complete["time_years"].to_numpy(), complete["died"].to_numpy()
    risk = {name: s.to_numpy() for name, s in recalibrated(complete, labels, load_preds("fused")).items()}

    tables, gaps, overall = [], [], []
    for name, r in risk.items():
        for attribute in REFERENCES:
            groups = labels[attribute]
            tables.append(subgroup_table(time, died, r, groups, args.n_boot).assign(variant=name, attribute=attribute))
            reference = reference_of(attribute, groups)
            for group in sorted(set(groups) - {reference}):
                in_a, in_b = (groups == group).to_numpy(), (groups == reference).to_numpy()
                gap, lo, hi = gap_ci(citl5, time, died, r, in_a, in_b, args.n_boot)
                row = {"variant": name, "attribute": attribute, "group": group, "versus": reference,
                       "metric": "citl5", "gap": gap, "gap_lo": lo, "gap_hi": hi}
                if name != "fused":
                    row["change"], row["change_lo"], row["change_hi"] = gap_change_ci(
                        citl5, time, died, r, risk["fused"], in_a, in_b, args.n_boot)
                gaps.append(row)
        row = {"variant": name}
        for metric_name, metric in METRICS.items():
            row[metric_name], row[f"{metric_name}_lo"], row[f"{metric_name}_hi"] = bootstrap_ci(
                metric, time, died, r, args.n_boot)
            if name != "fused":
                d, lo, hi = model_diff_ci(metric, time, died, r, risk["fused"], args.n_boot)
                row.update({f"{metric_name}_vs_fused": d, f"{metric_name}_vs_fused_lo": lo,
                            f"{metric_name}_vs_fused_hi": hi})
        overall.append(row)
        print("evaluated", name, flush=True)

    table = pd.concat(tables)
    table = table[["variant", "attribute"] + [c for c in table if c not in ("variant", "attribute")]]
    gaps, overall = pd.DataFrame(gaps), pd.DataFrame(overall)
    table.to_csv(RESULTS / "e5_groups.csv", index=False)
    gaps.to_csv(RESULTS / "e5_gaps.csv", index=False)
    overall.to_csv(RESULTS / "e5_overall.csv", index=False)
    print(gaps.round(4).to_string(index=False))
    print(overall.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

"""E2: missing data types. How much does each approach lose when a block is missing, simulated and real?

Simulated, on the complete-case out-of-fold predictions: at test time remove one or more blocks (see
CONDITIONS in crossfit.py) and compare four approaches:
- fused_naive: the fusion model trained on complete cases without modality dropout (missing blocks blanked)
- fused:       the fusion model trained with modality dropout (one model for every condition)
- cox_subset:  a Cox model refitted, on complete cases, on exactly the blocks that are still present
- cox_impute:  cox_full with the removed blocks treated as missing (mean-imputed, indicators on)
Real: people outside the complete-case set, grouped by what they actually lack, scored by fused and cox_full.
Writes results/e2_simulated.csv, e2_diffs.csv, e2_real.csv.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from nhanes_reliable.crossfit import CONDITIONS, crossfit, load_preds, oof_risk, prepare
from nhanes_reliable.folds import assign_folds
from nhanes_reliable.metrics import bootstrap_ci, brier5, c_index, citl5, model_diff_ci

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
APPROACHES = {"fused_naive": "fused_naive", "fused": "fused", "cox_subset": "cox_subset", "cox_impute": "cox_full"}
METRICS = {"c_index": c_index, "brier5": brier5, "citl5": citl5}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    cohort, days = prepare(ROOT / "data")
    folds = assign_folds(cohort)
    crossfit("fused_naive", cohort, days, folds, CONDITIONS, device=args.device, train_on="complete")
    crossfit("fused", cohort, days, folds, CONDITIONS, device=args.device)
    crossfit("cox_subset", cohort, days, folds, CONDITIONS, device=args.device, train_on="complete")
    crossfit("cox_full", cohort, days, folds, CONDITIONS, device=args.device)
    if args.skip_eval:
        return

    complete = cohort[cohort["complete_case"]]
    time, died = complete["time_years"].to_numpy(), complete["died"].to_numpy()
    preds = {a: load_preds(m) for a, m in APPROACHES.items()}
    risk = {(a, c): oof_risk(preds[a], c, complete.index) for a in APPROACHES for c in CONDITIONS}

    simulated, diffs = [], []
    for a in APPROACHES:
        for c in CONDITIONS:
            row = {"approach": a, "condition": c}
            for name, metric in METRICS.items():
                row[name], row[f"{name}_lo"], row[f"{name}_hi"] = bootstrap_ci(metric, time, died, risk[a, c],
                                                                              args.n_boot)
            simulated.append(row)
            contrasts = [] if c == "none" else [(f"{a} {c} - {a} none", risk[a, c], risk[a, "none"])]
            if a == "fused":
                contrasts += [(f"fused - fused_naive ({c})", risk[a, c], risk["fused_naive", c]),
                              (f"fused - cox_subset ({c})", risk[a, c], risk["cox_subset", c])]
            for label, ra, rb in contrasts:
                for name, metric in METRICS.items():
                    d, lo, hi = model_diff_ci(metric, time, died, ra, rb, args.n_boot)
                    diffs.append({"contrast": label, "approach": a, "condition": c, "metric": name,
                                  "diff": d, "lo": lo, "hi": hi})
        print("evaluated", a, flush=True)

    # Real missingness: the people outside the complete-case set, by what they lack.
    outside = cohort[~cohort["complete_case"]]
    groups = {
        "complete_case": cohort["complete_case"],
        "no_valid_wear": ~cohort["has_wearable"] & cohort["has_labs"] & cohort["has_exam"],
        "missing_labs": cohort["has_wearable"] & ~cohort["has_labs"],
    }
    groups["other"] = ~cohort["complete_case"] & ~groups["no_valid_wear"] & ~groups["missing_labs"]
    real = []
    for group, members in groups.items():
        people = cohort[members]
        base = {"group": group, "n": len(people), "deaths": int(people["died"].sum()),
                "deaths_5y": int(((people["died"] == 1) & (people["time_years"] < 5)).sum()),
                "mean_age": float(people["age"].mean())}
        if group == "other":
            real.append(base)  # too few people to score
            continue
        for model in ["fused", "cox_full"]:
            r = oof_risk(load_preds(model), "none", people.index)
            t, d = people["time_years"].to_numpy(), people["died"].to_numpy()
            row = {**base, "model": model}
            for name in ["c_index", "citl5"]:
                row[name], row[f"{name}_lo"], row[f"{name}_hi"] = bootstrap_ci(METRICS[name], t, d, r, args.n_boot)
            real.append(row)
    print(f"{len(outside)} people outside the complete-case set")

    simulated, diffs, real = pd.DataFrame(simulated), pd.DataFrame(diffs), pd.DataFrame(real)
    simulated.to_csv(RESULTS / "e2_simulated.csv", index=False)
    diffs.to_csv(RESULTS / "e2_diffs.csv", index=False)
    real.to_csv(RESULTS / "e2_real.csv", index=False)
    print(simulated.pivot_table(index="condition", columns="approach", values="c_index").round(3))
    print(real.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

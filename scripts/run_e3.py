"""E3: wear time. How do ranking and calibration change with fewer valid wear days?

Controlled: complete-case people with 7 valid days; at test time keep k = 1, 2, 3, 5, 7 of them (one fixed
random subset per person and k, shared by every model). The deep models are not retrained (their checkpoints
are loaded); cox_full is refitted per fold exactly as in E1 and scored with activity summaries recomputed from
the kept days.
Real: complete-case people grouped by their actual number of valid days (1-3, 4-6, 7).
Writes results/e3_days.csv, e3_diffs.csv, e3_real.csv.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from nhanes_reliable.activity import activity_summaries, keep_k_days
from nhanes_reliable.cox import COX_MODELS, cox_design, cox_risk, fit_cox
from nhanes_reliable.crossfit import CHECKPOINTS, fold_splits, load_preds, oof_risk, prepare
from nhanes_reliable.folds import assign_folds
from nhanes_reliable.metrics import bootstrap_ci, brier5, c_index, citl5, model_diff_ci
from nhanes_reliable.tabular import ACTIVITY_COLUMNS, fit_tab_stats, tab_variables
from nhanes_reliable.train import TrainConfig, build_fold_inputs, predict_risk, train_ensemble

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
KEEP = [1, 2, 3, 5, 7]
METRICS = {"c_index": c_index, "brier5": brier5, "citl5": citl5}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()

    cohort, days = prepare(ROOT / "data")
    folds = assign_folds(cohort)
    selected = (cohort["complete_case"] & (cohort["n_valid_days"] == 7)).to_numpy()
    kept = {k: keep_k_days(days.day_mask, k, seed=100 + k) for k in KEEP}
    time, died = cohort["time_years"].to_numpy(), cohort["died"].to_numpy()

    risk = {(m, k): np.full(len(cohort), np.nan) for m in ["fused", "deep_wearable", "cox_full"] for k in KEEP}
    for fold, test, fit, hold in fold_splits(cohort, folds):
        rows = test[selected[test]]
        for model in ["fused", "deep_wearable"]:
            inputs = build_fold_inputs(cohort, days, fit, model)
            members = train_ensemble(TrainConfig(model=model, device=args.device), inputs, fit, hold,
                                     CHECKPOINTS / model / f"fold{fold}")
            for k in KEEP:
                risk[model, k][rows] = predict_risk(members, inputs, rows, day_mask=kept[k], device=args.device)
        blocks = COX_MODELS["cox_full"]
        stats = fit_tab_stats(cohort.iloc[fit], tab_variables(blocks, with_activity=True))
        cox = fit_cox(cox_design(cohort, blocks, stats).iloc[fit], time[fit], died[fit])
        for k in KEEP:
            scored = cohort.iloc[rows].copy()
            scored[ACTIVITY_COLUMNS] = activity_summaries(days.x[rows], kept[k][rows])
            risk["cox_full", k][rows] = cox_risk(cox, cox_design(scored, blocks, stats))
        print(f"fold {fold} done", flush=True)

    t, d = time[selected], died[selected]
    rows, diffs = [], []
    for model in ["fused", "deep_wearable", "cox_full"]:
        for k in KEEP:
            r = risk[model, k][selected]
            row = {"model": model, "days_kept": k, "n": int(selected.sum()), "deaths": int(d.sum()),
                   "deaths_5y": int(((d == 1) & (t < 5)).sum())}
            for name, metric in METRICS.items():
                row[name], row[f"{name}_lo"], row[f"{name}_hi"] = bootstrap_ci(metric, t, d, r, args.n_boot)
            rows.append(row)
            if k != 7:
                for name, metric in METRICS.items():
                    diff, lo, hi = model_diff_ci(metric, t, d, r, risk[model, 7][selected], args.n_boot)
                    diffs.append({"model": model, "days_kept": k, "versus": 7, "metric": name,
                                  "diff": diff, "lo": lo, "hi": hi})
        print("evaluated", model, flush=True)

    complete = cohort[cohort["complete_case"]]
    bins = pd.cut(complete["n_valid_days"], [0, 3, 6, 7], labels=["1-3", "4-6", "7"])
    real = []
    for model in ["fused", "cox_full"]:
        r_all = pd.Series(oof_risk(load_preds(model), "none", complete.index), index=complete.index)
        for label in bins.cat.categories:
            people = complete[bins == label]
            r = r_all.loc[people.index].to_numpy()
            t, d = people["time_years"].to_numpy(), people["died"].to_numpy()
            row = {"model": model, "valid_days": label, "n": len(people), "deaths": int(d.sum()),
                   "deaths_5y": int(((d == 1) & (t < 5)).sum()),
                   "mean_age": float(people["age"].mean())}
            for name in ["c_index", "citl5"]:
                row[name], row[f"{name}_lo"], row[f"{name}_hi"] = bootstrap_ci(METRICS[name], t, d, r, args.n_boot)
            real.append(row)

    table, diffs, real = pd.DataFrame(rows), pd.DataFrame(diffs), pd.DataFrame(real)
    table.to_csv(RESULTS / "e3_days.csv", index=False)
    diffs.to_csv(RESULTS / "e3_diffs.csv", index=False)
    real.to_csv(RESULTS / "e3_real.csv", index=False)
    print(table.pivot_table(index="model", columns="days_kept", values="c_index").round(3))
    print(real.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

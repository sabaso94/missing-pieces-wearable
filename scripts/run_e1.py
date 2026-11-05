"""E1: baselines and fusion. How well does each model rank and calibrate 5-year mortality?

Six models, all cross-fitted over the same 5 folds and scored on the complete-case out-of-fold predictions:
- cox_agesex:    Cox on age and sex only (the floor; age dominates mortality)
- cox_clinical:  Cox on age, sex, labs, exam and questionnaire
- cox_full:      cox_clinical plus the 10 activity summaries
- deep_wearable: the day encoder on the minute-level wear data, plus age and sex
- deep_clinical: the tabular MLP on everything except the wearable
- fused:         wearable days + all tabular blocks, trained with modality and day dropout
Writes results/e1_overall.csv, e1_diffs.csv, e1_reliability.csv.
Use --models to train a subset (e.g. to spread the deep models over two GPUs); --skip-eval to only train.
"""

import argparse
from pathlib import Path

import pandas as pd

from nhanes_reliable.crossfit import crossfit, load_preds, oof_risk, prepare
from nhanes_reliable.folds import assign_folds
from nhanes_reliable.metrics import bootstrap_ci, brier5, c_index, citl5, model_diff_ci, reliability_table, uno_c

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MODELS = ["cox_agesex", "cox_clinical", "cox_full", "deep_wearable", "deep_clinical", "fused"]
METRICS = {"c_index": c_index, "uno_c": uno_c, "brier5": brier5, "citl5": citl5}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    cohort, days = prepare(ROOT / "data")
    folds = assign_folds(cohort)
    for model in args.models:
        crossfit(model, cohort, days, folds, {"none": []}, device=args.device)
    if args.skip_eval:
        return

    complete = cohort[cohort["complete_case"]]
    time, died = complete["time_years"].to_numpy(), complete["died"].to_numpy()
    risk = {m: oof_risk(load_preds(m), "none", complete.index) for m in MODELS}

    overall, diffs, reliability = [], [], []
    for m in MODELS:
        row = {"model": m}
        for name, metric in METRICS.items():
            row[name], row[f"{name}_lo"], row[f"{name}_hi"] = bootstrap_ci(metric, time, died, risk[m], args.n_boot)
        overall.append(row)
        for reference in ["cox_full", "cox_agesex"]:
            if m == reference:
                continue
            for name in ["c_index", "brier5", "citl5"]:
                d, lo, hi = model_diff_ci(METRICS[name], time, died, risk[m], risk[reference], args.n_boot)
                diffs.append({"model": m, "reference": reference, "metric": name, "diff": d, "lo": lo, "hi": hi})
        if m in ("cox_full", "fused"):
            reliability.append(reliability_table(time, died, risk[m]).assign(model=m))
        print("evaluated", m, flush=True)

    overall, diffs = pd.DataFrame(overall), pd.DataFrame(diffs)
    overall.to_csv(RESULTS / "e1_overall.csv", index=False)
    diffs.to_csv(RESULTS / "e1_diffs.csv", index=False)
    pd.concat(reliability).to_csv(RESULTS / "e1_reliability.csv", index=False)
    print(overall.round(4).to_string(index=False))
    print(diffs.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

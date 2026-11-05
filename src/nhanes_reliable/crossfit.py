"""Cross-fitting: for each of the 5 folds, fit on the other folds and predict the held-out one.

Inside each fold's training part, 20% is held out (for early stopping and recalibration). Predictions are saved
for both the test fold ("oof", out of fold) and the hold-out rows ("hold"), per missing-data condition.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .activity import DayData, activity_summaries, load_days
from .cox import COX_MODELS, cox_design, cox_risk, fit_cox
from .data import build_cohort, load_clinical, load_mortality
from .folds import inner_split
from .tabular import ACTIVITY_COLUMNS, fit_tab_stats, remove_blocks, tab_variables
from .train import DEEP_MODELS, TrainConfig, build_fold_inputs, predict_risk, train_ensemble

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINTS = ROOT / "checkpoints"
PREDS = ROOT / "results" / "preds"

CONDITIONS: dict[str, list[str]] = {
    "none": [],
    "no_wearable": ["wearable"],
    "no_labs": ["labs"],
    "no_exam": ["exam"],
    "no_questionnaire": ["questionnaire"],
    "no_clinic": ["labs", "exam"],                       # remote monitoring: wearable + questionnaire + age, sex
    "questionnaire_only": ["wearable", "labs", "exam"],  # questionnaire + age, sex
}


def prepare(data_dir: Path) -> tuple[pd.DataFrame, DayData]:
    """The cohort (with activity summaries as columns) and the day tensor aligned to its rows."""
    all_days = load_days(data_dir)
    n_valid = pd.Series(all_days.day_mask.sum(axis=1), index=pd.Index(all_days.seqn, name="seqn"))
    cohort = build_cohort(load_clinical(data_dir), load_mortality(data_dir), n_valid)
    rows = pd.Index(all_days.seqn).get_indexer(cohort.index)
    if (rows < 0).any():
        raise ValueError("cohort members missing from the minute-level subject list")
    days = DayData(seqn=cohort.index.to_numpy(), x=all_days.x[rows], day_mask=all_days.day_mask[rows])
    summaries = activity_summaries(days.x, days.day_mask)
    summaries[~days.day_mask.any(axis=1)] = np.nan  # no valid day: the whole wearable block is missing
    cohort[ACTIVITY_COLUMNS] = summaries
    return cohort, days


def fold_splits(cohort: pd.DataFrame, folds: pd.Series, train_on: str = "all"):
    """Yield (fold, test rows, fit rows, hold rows) as positions in `cohort`, the same for every script."""
    if train_on not in ("all", "complete"):
        raise ValueError(f"train_on must be 'all' or 'complete', got {train_on!r}")
    fold_of = folds.loc[cohort.index].to_numpy()
    died = cohort["died"].to_numpy()
    complete = cohort["complete_case"].to_numpy()
    for k in np.unique(fold_of):
        test = np.flatnonzero(fold_of == k)
        fit, hold = inner_split(np.flatnonzero(fold_of != k), died, fold=int(k))
        if train_on == "complete":
            fit, hold = fit[complete[fit]], hold[complete[hold]]
        yield int(k), test, fit, hold


def _cox_fold(model: str, cohort: pd.DataFrame, fit: np.ndarray, conditions: dict[str, list[str]],
              rows: np.ndarray) -> dict[str, np.ndarray]:
    """Risk for `rows` under each condition from Cox model(s) fitted on `fit`.

    Standard Cox models are fitted once and see removed blocks as missing (imputed, indicators on).
    "cox_subset" is refitted per condition on exactly the blocks that are still present.
    """
    time, died = cohort["time_years"].to_numpy(), cohort["died"].to_numpy()
    out, fitted = {}, None
    for condition, removed in conditions.items():
        if model == "cox_subset":
            blocks = [b for b in COX_MODELS["cox_full"] if b not in removed]
        else:
            blocks = COX_MODELS[model]
        if model == "cox_subset" or fitted is None:
            stats = fit_tab_stats(cohort.iloc[fit], tab_variables(blocks, with_activity=True))
            X = cox_design(cohort, blocks, stats)
            fitted = fit_cox(X.iloc[fit], time[fit], died[fit])
        X_rows = cox_design(remove_blocks(cohort.iloc[rows], removed), blocks, stats)
        out[condition] = cox_risk(fitted, X_rows)
    return out


def crossfit(model: str, cohort: pd.DataFrame, days: DayData | None, folds: pd.Series,
             conditions: dict[str, list[str]], device: str, train_on: str = "all",
             preds_dir: Path = PREDS, ckpt_root: Path = CHECKPOINTS) -> pd.DataFrame:
    """Out-of-fold and hold-out 5-year risks of one model under each condition; saved to preds_dir/<model>.csv.

    train_on="complete" fits only on complete cases (used for the naive deep model and the per-subset Cox).
    """
    records = []
    for k, test, fit, hold in fold_splits(cohort, folds, train_on):
        rows = np.r_[test, hold]
        if model in DEEP_MODELS:
            inputs = build_fold_inputs(cohort, days, fit, model)
            members = train_ensemble(TrainConfig(model=model, device=device), inputs, fit, hold,
                                     ckpt_root / model / f"fold{k}")
            risks = {c: predict_risk(members, inputs, rows, removed=r, device=device) for c, r in conditions.items()}
        else:
            risks = _cox_fold(model, cohort, fit, conditions, rows)
        split = np.r_[np.full(len(test), "oof"), np.full(len(hold), "hold")]
        for condition, risk in risks.items():
            records.append(pd.DataFrame({"seqn": cohort.index[rows], "fold": int(k), "split": split,
                                         "condition": condition, "risk5": risk}))
        print(f"{model}: fold {k} done", flush=True)
    preds = pd.concat(records, ignore_index=True)
    Path(preds_dir).mkdir(parents=True, exist_ok=True)
    preds.to_csv(Path(preds_dir) / f"{model}.csv", index=False)
    return preds


def load_preds(model: str, preds_dir: Path = PREDS) -> pd.DataFrame:
    return pd.read_csv(Path(preds_dir) / f"{model}.csv")


def oof_risk(preds: pd.DataFrame, condition: str, index: pd.Index) -> np.ndarray:
    """Out-of-fold 5-year risk under a condition, in the order of `index` (seqn)."""
    rows = preds[(preds["split"] == "oof") & (preds["condition"] == condition)].set_index("seqn")["risk5"]
    return rows.loc[index].to_numpy()

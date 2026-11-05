"""Cross-fitting folds: every person is in exactly one test fold, so every person gets one honest prediction.

With about a thousand deaths, a single 20% test set would hold about 200 and give wide intervals; pooling the
five out-of-fold predictions uses every death for evaluation.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def assign_folds(cohort: pd.DataFrame, n_folds: int = 5, seed: int = 0) -> pd.Series:
    """Fold 0..n_folds-1 per person, stratified by death x survey cycle."""
    strata = cohort["died"].astype(str) + "_" + cohort["cycle"].astype(str)
    folds = np.zeros(len(cohort), dtype=int)
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, test) in enumerate(splitter.split(np.zeros(len(cohort)), strata)):
        folds[test] = k
    return pd.Series(folds, index=cohort.index, name="fold")


def inner_split(train_idx: np.ndarray, died: np.ndarray, fold: int) -> tuple[np.ndarray, np.ndarray]:
    """Split a fold's training rows into (fit, hold): 20% held out, stratified by death.

    The hold-out set is used for early stopping and for fitting recalibration, never for fitting the model.
    """
    fit, hold = train_test_split(train_idx, test_size=0.2, stratify=died[train_idx], random_state=fold)
    return np.sort(fit), np.sort(hold)

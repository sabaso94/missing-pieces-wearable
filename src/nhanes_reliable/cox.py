"""Penalised Cox proportional-hazards baselines (lifelines) on the tabular features and activity summaries."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from .metrics import HORIZON
from .tabular import make_tab_features, tab_variables

COX_MODELS: dict[str, list[str]] = {
    "cox_agesex": ["core"],  # the floor: age dominates mortality
    "cox_clinical": ["core", "labs", "exam", "questionnaire"],
    "cox_full": ["core", "wearable", "labs", "exam", "questionnaire"],
}


@dataclass
class CoxModel:
    fitter: CoxPHFitter
    columns: list[str]  # the design columns the model was fitted on


def cox_design(cohort: pd.DataFrame, blocks: list[str], stats: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """Standardised variables of the blocks (activity summaries for the wearable block) plus missing indicators."""
    x, names = make_tab_features(cohort, stats, tab_variables(blocks, with_activity=True))
    return pd.DataFrame(x, columns=names, index=cohort.index)


def fit_cox(X: pd.DataFrame, time: np.ndarray, died: np.ndarray, penalizer: float = 0.1) -> CoxModel:
    """Fit a ridge-penalised Cox model.

    The penalty keeps coefficients finite when a rare 0/1 column (e.g. a missing indicator set for a few
    survivors only) nearly separates deaths from survivors; a weaker penalty (0.01) let the fit diverge.

    Columns that are constant in the fitting data, or exact copies of an earlier column, carry no information
    and are dropped. (All activity summaries are missing together, so their indicators are identical.)
    """
    X = X.loc[:, X.nunique() > 1]
    X = X.loc[:, ~X.T.duplicated()]
    data = X.assign(_time=np.asarray(time, dtype=float), _died=np.asarray(died, dtype=int))
    fitter = CoxPHFitter(penalizer=penalizer).fit(data, duration_col="_time", event_col="_died")
    return CoxModel(fitter=fitter, columns=list(X.columns))


def cox_risk(model: CoxModel, X: pd.DataFrame, years: float = HORIZON) -> np.ndarray:
    """Predicted risk of death before `years`: 1 - S(years- | x).

    S is a step function that jumps at the event times. lifelines interpolates linearly between them, so it is
    evaluated at the last event time before `years` (where interpolation is exact); deaths at `years` itself
    are not counted, as in the binary 5-year outcome.
    """
    event_times = model.fitter.baseline_survival_.index.to_numpy()
    before = event_times[event_times < years]
    if len(before) == 0:
        return np.zeros(len(X))
    survival = model.fitter.predict_survival_function(X[model.columns], times=[before[-1]])
    return 1 - survival.to_numpy()[0]

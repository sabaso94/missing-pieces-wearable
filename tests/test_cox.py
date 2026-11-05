import numpy as np
import pandas as pd

from nhanes_reliable.cox import COX_MODELS, cox_risk, fit_cox


def synthetic(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    t = rng.exponential(1 / (0.05 * np.exp(0.8 * X["x1"])))
    time, died = np.minimum(t, 9.0), (t < 9.0).astype(int)
    return X, time, died


def test_cox_models():
    assert COX_MODELS["cox_agesex"] == ["core"]
    assert COX_MODELS["cox_full"] == ["core", "wearable", "labs", "exam", "questionnaire"]


def test_cox_recovers_direction():
    X, time, died = synthetic()
    model = fit_cox(X, time, died)
    grid = pd.DataFrame({"x1": [-1.0, 0.0, 1.0], "x2": [0.0, 0.0, 0.0]})
    assert np.all(np.diff(cox_risk(model, grid)) > 0)


def test_constant_and_duplicate_columns_dropped():
    X, time, died = synthetic()
    X["always_zero"] = 0.0
    X["x2_copy"] = X["x2"]
    model = fit_cox(X, time, died)
    assert model.columns == ["x1", "x2"]
    assert np.isfinite(cox_risk(model, X)).all()


def test_risk_in_unit_interval():
    X, time, died = synthetic()
    risk = cox_risk(fit_cox(X, time, died), X)
    assert ((risk > 0) & (risk < 1)).all()


def test_cox_fit_part_only():
    X, time, died = synthetic()
    fit = np.arange(400)
    before = fit_cox(X.iloc[fit], time[fit], died[fit]).fitter.params_
    X2 = X.copy()
    X2.iloc[400:] = 99.0
    after = fit_cox(X2.iloc[fit], time[fit], died[fit]).fitter.params_
    pd.testing.assert_series_equal(before, after)

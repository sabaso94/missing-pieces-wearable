import numpy as np
from scipy.special import expit

from nhanes_reliable.causal import aipw_risk, e_value, overlap, risk_unadjusted


def _confounded(n=20_000, seed=0):
    """x0 raises both the chance of exposure and the risk; exposure lowers the risk. Returns the true RD too."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    t = (rng.random(n) < expit(1.0 * x[:, 0])).astype(int)
    risk = lambda tt: expit(-2 + 1.0 * x[:, 0] - 0.8 * tt)
    y = (rng.random(n) < risk(t)).astype(int)
    return y, t, x, float(np.mean(risk(1) - risk(0)))


def test_aipw_risk_recovers_a_known_risk_difference():
    y, t, x, true_rd = _confounded()
    assert abs(risk_unadjusted(y, t)["rd"] - true_rd) > 0.03
    out = aipw_risk(y, t, x, "linear")
    assert abs(out["rd"] - true_rd) < 0.01
    assert out["rd_low"] < true_rd < out["rd_high"] and out["rr_low"] < out["rr"] < out["rr_high"]
    assert abs(aipw_risk(y, t, x, "boosting")["rd"] - true_rd) < 0.015


def test_aipw_risk_survives_near_perfect_separation():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(600, 2))
    t = (x[:, 0] > 0).astype(int)
    t[:4] = 1 - t[:4]
    y = (rng.random(600) < 0.2).astype(int)
    for nuisance in ["linear", "boosting"]:
        out = aipw_risk(y, t, x, nuisance)
        assert all(np.isfinite(out[k]) for k in ["rd", "rd_low", "rd_high", "rr", "rr_low", "rr_high"])


def test_e_value_of_a_protective_risk_ratio():
    assert abs(e_value(0.5) - (2 + np.sqrt(2))) < 1e-9
    assert abs(e_value(2.0) - (2 + np.sqrt(2))) < 1e-9
    assert e_value(1.0) == 1.0


def test_unadjusted_risks_are_group_means():
    y, t = np.array([1, 0, 0, 0, 1, 1]), np.array([1, 1, 1, 1, 0, 0])
    out = risk_unadjusted(y, t)
    assert out["risk1"] == 0.25 and out["risk0"] == 1.0 and out["rd"] == -0.75 and out["rr"] == 0.25


def test_overlap_counts_extreme_propensities():
    out = overlap(np.array([0.01, 0.04, 0.5, 0.96, 0.99]))
    assert out == {"below_0.05": 0.4, "above_0.95": 0.4}


def test_aipw_risk_can_return_the_propensities():
    y, t, x, _ = _confounded(n=2000)
    out, e = aipw_risk(y, t, x, "linear", return_propensity=True)
    assert e.shape == (2000,) and e.min() >= 0.01 and e.max() <= 0.99
    assert out == aipw_risk(y, t, x, "linear")

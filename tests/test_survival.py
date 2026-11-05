import math

import numpy as np
import pytest
import torch
from lifelines import KaplanMeierFitter

from nhanes_reliable.survival import N_INTERVALS, discretise, km_censoring, km_risk, logistic_hazard_loss, risk_at


def test_discretise_boundaries():
    assert N_INTERVALS == 10
    assert discretise(np.array([0.0, 59 / 12, 5.0, 9.99])).tolist() == [0, 4, 5, 9]
    with pytest.raises(ValueError):
        discretise(np.array([10.0]))
    with pytest.raises(ValueError):
        discretise(np.array([-0.1]))


def test_loss_hand_computed():
    logits = torch.zeros(2, N_INTERVALS)  # every hazard 0.5, so every at-risk term is ln 2
    loss = logistic_hazard_loss(logits, torch.tensor([2, 0]), torch.tensor([1.0, 0.0]))
    assert loss.item() == pytest.approx(math.log(2))  # 3 terms from the death, none from the early censoring
    with pytest.raises(ValueError):
        logistic_hazard_loss(logits[:1], torch.tensor([0]), torch.tensor([0.0]))


def test_loss_targets():
    # Death in interval 1 with confident logits: low hazard in interval 0, high in 1 gives a small loss.
    logits = torch.full((1, N_INTERVALS), -10.0)
    logits[0, 1] = 10.0
    assert logistic_hazard_loss(logits, torch.tensor([1]), torch.tensor([1.0])).item() < 1e-3
    # A person censored in interval 3 contributes intervals 0-2 only, so a high hazard in 3 costs nothing.
    logits = torch.full((1, N_INTERVALS), -10.0)
    logits[0, 3] = 10.0
    assert logistic_hazard_loss(logits, torch.tensor([3]), torch.tensor([0.0])).item() < 1e-3


def test_risk_at():
    h = np.full((2, N_INTERVALS), 0.1)
    assert risk_at(h, 5)[0] == pytest.approx(1 - 0.9**5)
    assert (np.diff([risk_at(h, y)[0] for y in range(1, N_INTERVALS + 1)]) >= 0).all()


def test_risk_matches_binary_outcome():
    time = np.array([0.0, 1.5, 4.99, 5.0, 6.0, 8.0, 7.5, 9.0])
    died = np.array([1, 1, 1, 1, 0, 0, 1, 0])
    assert km_risk(time, died, 5) == pytest.approx(3 / 8)  # the death at exactly 5.0 is not within 5 years


def test_km_matches_lifelines():
    rng = np.random.default_rng(0)
    time = rng.exponential(5, 300).round(2)
    died = rng.random(300) < 0.6
    kmf = KaplanMeierFitter().fit(time, died)
    for t in [1.234, 3.333, 6.789]:  # times without events exactly at t
        assert km_risk(time, died, t) == pytest.approx(1 - kmf.predict(t), abs=1e-10)


def test_km_censoring_swaps_roles():
    time = np.array([1.0, 2.0, 3.0, 4.0])
    died = np.array([1, 0, 1, 0])
    G = km_censoring(time, died)
    assert G(np.array([1.5]))[0] == 1.0  # nobody censored before 1.5
    assert G(np.array([2.5]))[0] == pytest.approx(2 / 3)  # one of the three still at risk at 2.0 is censored
    assert G(np.array([2.0]), left=True)[0] == 1.0

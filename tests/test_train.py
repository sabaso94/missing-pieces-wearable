import numpy as np
import pandas as pd
import torch

from nhanes_reliable.activity import DayData
from nhanes_reliable.data import BLOCKS, REMOVABLE
from nhanes_reliable.metrics import c_index
from nhanes_reliable.train import (
    DEEP_MODELS,
    TrainConfig,
    apply_flags,
    build_fold_inputs,
    predict_risk,
    sample_block_dropout,
    sample_day_dropout,
    train_ensemble,
    train_model,
)


def synthetic_cohort(n=300, seed=0):
    """Cohort whose hazard rises with hba1c; some people have no valid wear day or no labs."""
    rng = np.random.default_rng(seed)
    cohort = pd.DataFrame({v: rng.normal(size=n) for b in BLOCKS.values() for v in b},
                          index=pd.Index(np.arange(n), name="seqn"))
    t = rng.exponential(1 / (0.1 * np.exp(1.5 * cohort["hba1c"].to_numpy())))
    cohort["time_years"], cohort["died"] = np.minimum(t, 9.5), (t < 9.5).astype(int)
    x = np.zeros((n, 7, 2, 1440), np.float32)
    x[:, :, 0] = rng.uniform(0, 2, (n, 7, 1440))
    x[:, :, 1] = 1.0
    mask = np.ones((n, 7), bool)
    mask[:20] = False
    mask[20:60, 3:] = False
    cohort.loc[cohort.index[-15:], BLOCKS["labs"]] = np.nan
    cohort["n_valid_days"] = mask.sum(axis=1)
    cohort["has_wearable"] = cohort["n_valid_days"] > 0
    for b in ["labs", "exam", "questionnaire"]:
        cohort[f"has_{b}"] = cohort[BLOCKS[b]].notna().any(axis=1)
    return cohort, DayData(seqn=cohort.index.to_numpy(), x=x, day_mask=mask)


def test_deep_models():
    assert set(DEEP_MODELS) == {"fused", "fused_naive", "deep_wearable", "deep_clinical"}
    assert DEEP_MODELS["fused"]["modality_dropout"] and not DEEP_MODELS["fused_naive"]["modality_dropout"]
    assert not DEEP_MODELS["deep_clinical"]["use_wearable"]


def test_block_dropout_only_turns_off():
    g = torch.Generator().manual_seed(0)
    flags = torch.tensor([[1.0, 0.0, 1.0, 1.0]]).repeat(2000, 1)
    dropped = sample_block_dropout(flags, 0.3, g)
    assert (dropped <= flags).all() and (dropped[:, 1] == 0).all()
    assert abs(1 - dropped[:, 0].mean().item() - 0.3) < 0.04


def test_apply_flags_blanks_block_like_missing():
    cohort, days = synthetic_cohort()
    inputs = build_fold_inputs(cohort, days, np.arange(200), "fused")
    tab = torch.from_numpy(inputs.tab[:5]).clone()
    mask = torch.from_numpy(days.day_mask[30:35]).clone()
    flags = torch.ones(5, 4)
    flags[:, REMOVABLE.index("labs")] = 0
    flags[:, REMOVABLE.index("wearable")] = 0
    tab2, mask2 = apply_flags(tab, mask, flags, inputs)
    labs = inputs.tab_blocks == "labs"
    assert (tab2[:, labs & ~inputs.tab_indicator] == 0).all() and (tab2[:, labs & inputs.tab_indicator] == 1).all()
    torch.testing.assert_close(tab2[:, ~labs], tab[:, ~labs])
    assert not mask2.any()
    # A block removed this way looks exactly like a block that was missing in the data.
    missing_row = inputs.tab[-1]  # the last people have no labs
    assert (missing_row[labs & inputs.tab_indicator] == 1).all() and (missing_row[labs & ~inputs.tab_indicator] == 0).all()


def test_day_dropout_keeps_at_least_one():
    g = torch.Generator().manual_seed(0)
    mask = torch.zeros(500, 7, dtype=torch.bool)
    mask[:, :5] = True
    mask[:100] = False
    kept = sample_day_dropout(mask, g)
    assert (kept <= mask).all()
    assert (kept[100:].sum(dim=1) >= 1).all() and not kept[:100].any()
    assert kept[100:].sum(dim=1).float().mean() < 5  # some days are dropped


def test_fold_stats_fit_part_only():
    cohort, days = synthetic_cohort()
    fit = np.arange(200)
    before = build_fold_inputs(cohort, days, fit, "fused")
    days.x[250:, :, 0] = 50.0
    cohort.iloc[250:, cohort.columns.get_loc("hba1c")] = 1e3
    after = build_fold_inputs(cohort, days, fit, "fused")
    assert before.mims_mean == after.mims_mean and before.mims_std == after.mims_std
    np.testing.assert_array_equal(before.tab[:200], after.tab[:200])


def test_training_learns_signal():
    cohort, days = synthetic_cohort(n=1500)  # an oracle scoring hba1c reaches C = 0.80 on this data
    fit, hold = np.arange(1000), np.arange(1000, 1485)  # the last 15 people have no labs
    inputs = build_fold_inputs(cohort, days, fit, "deep_clinical")
    cfg = TrainConfig(model="deep_clinical", device="cpu", max_epochs=15, patience=15, lr=3e-3)
    model, history = train_model(cfg, inputs, fit, hold)
    assert history["loss"].iloc[-1] < history["loss"].iloc[0]
    risk = predict_risk([model], inputs, hold, device="cpu")
    assert c_index(inputs.time[hold], inputs.died[hold], risk) > 0.7


def test_checkpoint_reload_identical(tmp_path):
    cohort, days = synthetic_cohort()
    fit, hold = np.arange(200), np.arange(200, 300)
    inputs = build_fold_inputs(cohort, days, fit, "fused")
    cfg = TrainConfig(model="fused", device="cpu", max_epochs=2, patience=2)
    first = train_ensemble(cfg, inputs, fit, hold, tmp_path, seeds=range(2))
    again = train_ensemble(cfg, inputs, fit, hold, tmp_path, seeds=range(2))
    np.testing.assert_array_equal(predict_risk(first, inputs, hold, device="cpu"),
                                  predict_risk(again, inputs, hold, device="cpu"))


def test_removed_blocks_change_prediction_but_stay_valid():
    cohort, days = synthetic_cohort()
    fit, hold = np.arange(200), np.arange(200, 300)
    inputs = build_fold_inputs(cohort, days, fit, "fused")
    model, _ = train_model(TrainConfig(model="fused", device="cpu", max_epochs=2, patience=2), inputs, fit, hold)
    full = predict_risk([model], inputs, hold, device="cpu")
    no_labs = predict_risk([model], inputs, hold, removed=["labs"], device="cpu")
    assert ((no_labs > 0) & (no_labs < 1)).all() and not np.allclose(full, no_labs)

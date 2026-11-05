import numpy as np
import pandas as pd
import pytest

from nhanes_reliable import activity
from nhanes_reliable.activity import (
    MAX_DAYS,
    MIN_WEAR_MINUTES,
    SUMMARY_NAMES,
    DayData,
    activity_summaries,
    keep_k_days,
    load_days,
    select_valid_days,
    valid_day,
)


def test_valid_day_rule():
    worn = np.zeros((2, 1440), dtype=bool)
    worn[0, :MIN_WEAR_MINUTES] = True
    worn[1, : MIN_WEAR_MINUTES - 1] = True
    assert MIN_WEAR_MINUTES == 1368
    assert list(valid_day(worn)) == [True, False]


def test_first_seven_valid_days():
    # One person, 9 recorded days; day 3 is invalid, so the first seven valid days are 1, 2, 4, 5, 6, 7, 8.
    log_mims = np.arange(9, dtype=np.float32)[:, None].repeat(1440, axis=1)
    worn = np.ones((9, 1440), dtype=bool)
    worn[2, :100] = False
    days = select_valid_days(np.full(9, 7), np.arange(1, 10), log_mims, worn, seqn=np.array([7]))
    assert days.day_mask.tolist() == [[True] * 7]
    assert days.x[0, :, 0, 0].tolist() == [0, 1, 3, 4, 5, 6, 7]
    assert days.x.shape == (1, MAX_DAYS, 2, 1440)


def test_non_worn_minutes_zeroed_and_flagged():
    log_mims = np.full((1, 1440), 2.0, dtype=np.float32)
    worn = np.ones((1, 1440), dtype=bool)
    worn[0, :10] = False
    days = select_valid_days(np.array([1]), np.array([2]), log_mims, worn, seqn=np.array([1]))
    assert (days.x[0, 0, 0, :10] == 0).all() and (days.x[0, 0, 1, :10] == 0).all()
    assert (days.x[0, 0, 0, 10:] == 2).all() and (days.x[0, 0, 1, 10:] == 1).all()


def person(values_per_day: list[np.ndarray], n_days: int = MAX_DAYS) -> tuple[np.ndarray, np.ndarray]:
    """(1, 7, 2, 1440) tensor from per-day log10(1 + MIMS) profiles, fully worn; later days masked."""
    x = np.zeros((1, MAX_DAYS, 2, 1440), dtype=np.float32)
    mask = np.zeros((1, MAX_DAYS), dtype=bool)
    for d, values in enumerate(values_per_day[:n_days]):
        x[0, d, 0], x[0, d, 1], mask[0, d] = values, 1.0, True
    return x, mask


def test_summaries_ignore_invalid_days():
    rng = np.random.default_rng(0)
    x, mask = person([rng.uniform(0, 2, 1440).astype(np.float32) for _ in range(7)])
    mask[0, 5:] = False
    before = activity_summaries(x, mask)
    x[0, 5:, 0] = 3.0
    np.testing.assert_allclose(activity_summaries(x, mask), before)


def test_summaries_constant_signal():
    x, mask = person([np.full(1440, 1.5, dtype=np.float32)] * 4)
    s = dict(zip(SUMMARY_NAMES, activity_summaries(x, mask)[0]))
    assert s["astp"] == 0 and s["satp"] == 0
    assert s["relative_amplitude"] == pytest.approx(0)
    assert s["n_valid_days"] == 4
    assert s["total_mims"] == pytest.approx(1440 * (10**1.5 - 1), rel=1e-5)
    assert s["active_minutes"] == 1440  # 10**1.5 - 1 = 30.6 MIMS/min is above the 10.558 threshold


def test_transition_probabilities():
    # Alternating 60 active / 60 inactive minutes: every hour one transition each way.
    hi, lo = np.log10(1 + 50.0), 0.0
    day = np.tile(np.r_[np.full(60, hi), np.full(60, lo)], 12).astype(np.float32)
    s = dict(zip(SUMMARY_NAMES, activity_summaries(*person([day]))[0]))
    assert s["astp"] == pytest.approx(12 / 720, rel=1e-6)  # 12 active->inactive transitions over 720 active minutes
    assert s["satp"] == pytest.approx(11 / 719, rel=1e-6)  # the last inactive minute has no successor in the day


def test_no_valid_days_nan():
    x, mask = person([], n_days=0)
    assert np.isnan(activity_summaries(x, mask)[0, : SUMMARY_NAMES.index("n_valid_days")]).all()
    assert activity_summaries(x, mask)[0, SUMMARY_NAMES.index("n_valid_days")] == 0


def test_keep_k_days():
    mask = np.zeros((3, MAX_DAYS), dtype=bool)
    mask[0, :7], mask[1, :2], mask[2, [0, 3, 5]] = True, True, True
    kept = keep_k_days(mask, k=3, seed=1)
    assert kept.sum(axis=1).tolist() == [3, 2, 3]
    assert not (kept & ~mask).any()
    np.testing.assert_array_equal(kept, keep_k_days(mask, k=3, seed=1))


def test_cache_mismatch_rebuilds(tmp_path, monkeypatch):
    (tmp_path / "physionet").mkdir()
    pd.DataFrame({"SEQN": [1, 2]}).to_csv(tmp_path / "physionet" / "subject-info.csv", index=False)
    calls = []

    def fake_build(data_dir, seqn):
        calls.append(seqn.tolist())
        return DayData(seqn=seqn, x=np.zeros((len(seqn), MAX_DAYS, 2, 1440), np.float32),
                       day_mask=np.zeros((len(seqn), MAX_DAYS), bool))

    monkeypatch.setattr(activity, "_build_days", fake_build)
    load_days(tmp_path)  # no cache: builds
    load_days(tmp_path)  # cache matches: no rebuild
    pd.DataFrame({"SEQN": [1, 2, 3]}).to_csv(tmp_path / "physionet" / "subject-info.csv", index=False)
    days = load_days(tmp_path)  # subjects changed: rebuilds
    assert calls == [[1, 2], [1, 2, 3]] and days.seqn.tolist() == [1, 2, 3]

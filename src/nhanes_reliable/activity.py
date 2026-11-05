"""Minute-level wrist activity: valid days, the day tensor for the deep model, and summaries for the Cox model.

Source: PhysioNet "Minute-level step counts and physical activity data from NHANES 2011-2014" (v1.0.2).
One row per person-day, 1,440 minute columns. Activity is MIMS (monitor-independent movement summary),
stored as log10(1 + MIMS). A minute counts as worn when the NHANES wear prediction (PAXPREDM) says wake wear
(1) or sleep wear (2) and the quality flag (PAXFLGSM) is FALSE.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

MINUTES = 1440
MAX_DAYS = 7
# A day is valid if at least 95% of its minutes are worn and unflagged (Leroux et al. 2019). This also drops
# the partial first and last recording days.
MIN_WEAR_MINUTES = 1368
# A minute with MIMS >= 10.558 counts as active (Karas et al. 2022).
ACTIVE_MIMS = 10.558
WORN_CODES = (1, 2)  # PAXPREDM: 1 wake wear, 2 sleep wear

SUMMARY_NAMES: list[str] = [
    "total_mims",              # mean daily sum of MIMS
    "active_minutes",          # mean daily minutes with MIMS >= ACTIVE_MIMS
    "peak30_mims",             # mean daily average of the 30 highest minutes
    "astp",                    # P(next minute inactive | active minute), pooled over days
    "satp",                    # P(next minute active | inactive minute)
    "m10",                     # mean MIMS of the most active 10 consecutive hours of the average day
    "l5",                      # mean MIMS of the least active 5 consecutive hours
    "relative_amplitude",      # (M10 - L5) / (M10 + L5)
    "intradaily_variability",  # fragmentation of the hourly series
    "n_valid_days",
]


@dataclass
class DayData:
    seqn: np.ndarray      # (N,) NHANES ids
    x: np.ndarray         # (N, 7, 2, 1440) float32: channel 0 log10(1 + MIMS) (0 where not worn), channel 1 worn
    day_mask: np.ndarray  # (N, 7) bool: which of the 7 slots hold a valid day


def valid_day(worn: np.ndarray) -> np.ndarray:
    """(R, 1440) worn minutes -> (R,) bool: at least MIN_WEAR_MINUTES worn."""
    return worn.sum(axis=1) >= MIN_WEAR_MINUTES


def select_valid_days(row_seqn: np.ndarray, row_day: np.ndarray, log_mims: np.ndarray, worn: np.ndarray,
                      seqn: np.ndarray) -> DayData:
    """Keep each person's first MAX_DAYS valid days, in recording order, as a DayData aligned to `seqn`."""
    order = np.lexsort((row_day, row_seqn))
    row_seqn, log_mims, worn = row_seqn[order], log_mims[order], worn[order]
    keep = valid_day(worn)
    # Rank of each valid day within its person: 0, 1, 2, ...
    rank = pd.Series(1, index=row_seqn[keep]).groupby(level=0).cumcount().to_numpy()
    rows = np.flatnonzero(keep)[rank < MAX_DAYS]
    slots = rank[rank < MAX_DAYS]
    people = pd.Index(seqn).get_indexer(row_seqn[rows])
    if (people < 0).any():
        raise ValueError("minute data contains SEQNs that are not in the subject list")

    x = np.zeros((len(seqn), MAX_DAYS, 2, MINUTES), dtype=np.float32)
    day_mask = np.zeros((len(seqn), MAX_DAYS), dtype=bool)
    x[people, slots, 0] = np.where(worn[rows], np.nan_to_num(log_mims[rows]), 0.0)
    x[people, slots, 1] = worn[rows]
    day_mask[people, slots] = True
    return DayData(seqn=np.asarray(seqn), x=x, day_mask=day_mask)


def _read_minute_file(path: Path, flags: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(row SEQN, row day number, (R, 1440) values) from one nhanes_1440_<var>.csv.xz file."""
    if flags:
        table = pd.read_csv(path, true_values=["TRUE"], false_values=["FALSE"], na_values=["NA"])
        values = table.iloc[:, 3:].eq(False).to_numpy()  # True only where explicitly unflagged
    else:
        table = pd.read_csv(path, na_values=["NA"], dtype={f"min_{m:04d}": np.float32 for m in range(1, 1441)})
        values = table.iloc[:, 3:].to_numpy(dtype=np.float32)
    return table["SEQN"].to_numpy(dtype=int), table["PAXDAYM"].to_numpy(dtype=int), values


def _build_days(data_dir: Path, seqn: np.ndarray) -> DayData:
    folder = data_dir / "physionet"
    row_seqn, row_day, log_mims = _read_minute_file(folder / "nhanes_1440_log10PAXMTSM.csv.xz")
    s2, d2, prediction = _read_minute_file(folder / "nhanes_1440_PAXPREDM.csv.xz")
    s3, d3, unflagged = _read_minute_file(folder / "nhanes_1440_PAXFLGSM.csv.xz", flags=True)
    if not (np.array_equal(row_seqn, s2) and np.array_equal(row_seqn, s3)
            and np.array_equal(row_day, d2) and np.array_equal(row_day, d3)):
        raise ValueError("the minute-level files are not in the same row order")
    worn = np.isin(prediction, WORN_CODES) & unflagged & ~np.isnan(log_mims)
    return select_valid_days(row_seqn, row_day, log_mims, worn, seqn)


def load_days(data_dir: Path) -> DayData:
    """Day tensor for every subject in subject-info.csv (people without valid days get an empty mask).

    Cached at data/cache/days.npz together with the SEQNs; rebuilt if the subject list changed.
    """
    seqn = np.sort(pd.read_csv(data_dir / "physionet" / "subject-info.csv")["SEQN"].to_numpy(dtype=int))
    cache = data_dir / "cache" / "days.npz"
    if cache.exists():
        stored = np.load(cache)
        if np.array_equal(stored["seqn"], seqn):
            return DayData(seqn=stored["seqn"], x=stored["x"], day_mask=stored["day_mask"])
    days = _build_days(data_dir, seqn)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, seqn=days.seqn, x=days.x, day_mask=days.day_mask)
    return days


def n_valid_days(days: DayData) -> pd.Series:
    return pd.Series(days.day_mask.sum(axis=1), index=pd.Index(days.seqn, name="seqn"))


def _transition_probability(state: np.ndarray, valid: np.ndarray, start: bool) -> float:
    """P(state flips | current state == start), counted within days; minutes without a successor are skipped."""
    current, following = state[valid, :-1], state[valid, 1:]
    at_risk = current == start
    return float((at_risk & (following != start)).sum() / at_risk.sum()) if at_risk.any() else 0.0


def _intradaily_variability(hourly: np.ndarray) -> float:
    """IV = n * sum of squared successive differences / ((n - 1) * sum of squared deviations); 0 if constant."""
    deviations = ((hourly - hourly.mean()) ** 2).sum()
    if len(hourly) < 2 or deviations == 0:
        return 0.0
    return float(len(hourly) * (np.diff(hourly) ** 2).sum() / ((len(hourly) - 1) * deviations))


def activity_summaries(x: np.ndarray, day_mask: np.ndarray) -> np.ndarray:
    """(N, len(SUMMARY_NAMES)) summaries from the valid days only; NaN (except n_valid_days = 0) without any."""
    out = np.full((len(x), len(SUMMARY_NAMES)), np.nan)
    out[:, -1] = day_mask.sum(axis=1)
    mims = np.power(10.0, x[:, :, 0].astype(np.float64)) - 1  # non-worn minutes were stored as 0 -> 0 MIMS
    for i in np.flatnonzero(day_mask.any(axis=1)):
        days = mims[i, day_mask[i]]
        active = days >= ACTIVE_MIMS
        everything = np.ones(len(days), dtype=bool)
        hourly_profile = days.mean(axis=0).reshape(24, 60).mean(axis=1)
        circular = np.r_[hourly_profile, hourly_profile]
        m10 = max(circular[h : h + 10].mean() for h in range(24))
        l5 = min(circular[h : h + 5].mean() for h in range(24))
        out[i, :-1] = [
            days.sum(axis=1).mean(),
            active.sum(axis=1).mean(),
            np.sort(days, axis=1)[:, -30:].mean(),
            _transition_probability(active, everything, start=True),
            _transition_probability(active, everything, start=False),
            m10,
            l5,
            (m10 - l5) / (m10 + l5) if m10 + l5 > 0 else 0.0,
            _intradaily_variability(days.reshape(len(days), 24, 60).mean(axis=2).ravel()),
        ]
    return out


def keep_k_days(day_mask: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Keep a fixed random subset of min(k, n_valid) valid days per person (same seed, same subset)."""
    keys = np.random.default_rng(seed).random(day_mask.shape)
    keys[~day_mask] = np.inf
    rank = keys.argsort(axis=1).argsort(axis=1)
    return day_mask & (rank < k)

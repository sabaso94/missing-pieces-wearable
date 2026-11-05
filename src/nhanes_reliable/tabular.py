"""Tabular features: standardised values plus missing indicators, and removal of whole blocks.

The activity summaries (for the Cox models) live in the cohort frame as ACTIVITY_COLUMNS and belong to the
wearable block; the deep models see the wearable block as the day tensor instead.
"""

import numpy as np
import pandas as pd

from .activity import SUMMARY_NAMES
from .data import BLOCKS, REMOVABLE

ACTIVITY_COLUMNS: list[str] = [f"act_{name}" for name in SUMMARY_NAMES]


def tab_variables(blocks: list[str], with_activity: bool = False) -> list[str]:
    """Tabular variables of the named blocks, in BLOCKS order (plus the activity summaries if asked)."""
    variables = [v for block in BLOCKS for v in BLOCKS[block] if block in blocks]
    if with_activity and "wearable" in blocks:
        variables += ACTIVITY_COLUMNS
    return variables


def fit_tab_stats(df: pd.DataFrame, variables: list[str]) -> dict[str, tuple[float, float]]:
    """Mean and std of each variable over its non-missing values (pass only the fitting rows)."""
    stats = {}
    for v in variables:
        mean, std = float(df[v].mean()), float(df[v].std())
        stats[v] = (mean, std if std > 0 else 1.0)
    return stats


def make_tab_features(df: pd.DataFrame, stats: dict[str, tuple[float, float]],
                      variables: list[str]) -> tuple[np.ndarray, list[str]]:
    """(N, 2V) float32: standardised values with missing set to 0 (the training mean), then 0/1 indicators."""
    values = np.column_stack([(df[v].to_numpy(dtype=float) - stats[v][0]) / stats[v][1] for v in variables])
    missing = np.isnan(values)
    x = np.concatenate([np.where(missing, 0.0, values), missing.astype(float)], axis=1).astype(np.float32)
    return x, variables + [f"{v}_missing" for v in variables]


def remove_blocks(df: pd.DataFrame, blocks: list[str]) -> pd.DataFrame:
    """Copy with every variable of the named blocks set to missing (core can never be removed)."""
    unknown = [b for b in blocks if b not in REMOVABLE]
    if unknown:
        raise ValueError(f"cannot remove {unknown}; removable blocks are {REMOVABLE}")
    out = df.copy()
    columns = tab_variables(blocks, with_activity=True)
    out[[c for c in columns if c in out]] = np.nan
    return out


def block_flags(cohort: pd.DataFrame, removed: list[str] = ()) -> np.ndarray:
    """(N, 4) float32 block-present flags in REMOVABLE order; removed blocks are forced to 0."""
    flags = np.column_stack([cohort[f"has_{b}"].to_numpy(dtype=float) for b in REMOVABLE])
    for b in removed:
        flags[:, REMOVABLE.index(b)] = 0.0
    return flags.astype(np.float32)

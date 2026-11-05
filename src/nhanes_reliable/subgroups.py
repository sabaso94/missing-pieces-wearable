"""Subgroups: age bands, and per-group metrics with person-level bootstrap CIs on the gaps between groups."""

import numpy as np
import pandas as pd

AGE_EDGES = (20, 50, 65)  # bands 20-49, 50-64, 65+ (NHANES top-codes age at 80)


def age_band(age: pd.Series, edges: tuple[int, ...] = AGE_EDGES) -> pd.Series:
    """Ordered categorical age band, labelled like "20-49" and "65+"."""
    labels = [f"{a}-{b - 1}" for a, b in zip(edges[:-1], edges[1:])] + [f"{edges[-1]}+"]
    return pd.cut(age, bins=list(edges) + [np.inf], right=False, labels=labels, ordered=True)

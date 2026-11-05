import pandas as pd

from nhanes_reliable.subgroups import age_band


def test_age_band_edges():
    bands = age_band(pd.Series([20.0, 49.0, 50.0, 64.0, 65.0, 80.0]))
    assert bands.astype(str).tolist() == ["20-49", "20-49", "50-64", "50-64", "65+", "65+"]
    assert list(bands.cat.categories) == ["20-49", "50-64", "65+"] and bands.cat.ordered

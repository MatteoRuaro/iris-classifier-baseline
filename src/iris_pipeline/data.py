from __future__ import annotations
import pandas as pd
from sklearn.datasets import load_iris


def load_data() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    ds = load_iris(as_frame=True)
    X: pd.DataFrame = ds.data
    y: pd.Series = ds.target
    feature_names = list(ds.feature_names)
    return X, y, feature_names

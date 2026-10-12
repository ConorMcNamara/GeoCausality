"""Tests for the cross-validation helpers in ``GeoCausality.utils``."""

import pandas as pd

from GeoCausality.utils import HoldoutSplitter


def test_holdout_splitter_drops_by_position_with_duplicate_index() -> None:
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]}, index=[0, 0, 1, 1])
    ser = pd.Series([10.0, 20.0, 30.0, 40.0], index=df.index)
    for i, (x_train, x_holdout, y_train, y_holdout) in enumerate(HoldoutSplitter(df, ser, 1)):
        assert x_train.shape[0] == y_train.shape[0] == 3
        assert x_holdout["x"].tolist() == [df["x"].iloc[i]]
        assert y_holdout.tolist() == [ser.iloc[i]]

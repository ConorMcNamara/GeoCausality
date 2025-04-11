import pandas as pd


class HoldoutSplitter:
    """Iterator that prepares the time series for cross-validation by
    progressively removing blocks of length `holdout_len`.
    """

    def __init__(self, df: pd.DataFrame, ser: pd.Series, holdout_len: int = 1):
        """Iterator that prepares the time series for cross-validation by
        progressively removing blocks of length `holdout_len`.

        Parameters
        ----------
        df : pandas.DataFrame, shape (r, c)
            Dataframe that will be split for the cross-validation.
        ser : pandas.Series, shape (r, 1)
            Series that will split for the cross-validation.
        holdout_len : int, optional
            Number of days to remove in each iteration, by default 1.

        Raises
        ------
        ValueError
            if df and ser do not have the same number of rows.
        ValueError
            if `holdout_len` is not >= 1.
        ValueError
            if `holdout_len` is larger than the number of rows of df.
        """
        if df.shape[0] != ser.shape[0]:
            raise ValueError("`df` and `ser` must have the same number of rows.")
        if holdout_len < 1:
            raise ValueError("`holdout_len` must be at least 1.")
        if holdout_len >= df.shape[0]:
            raise ValueError("`holdout_len` must be less than df.shape[0]")
        self.df = df
        self.ser = ser
        self.holdout_len = holdout_len
        self.idx = 0

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        if (self.idx + self.holdout_len) > self.df.shape[0]:
            raise StopIteration
        holdout = slice(self.idx, self.idx + self.holdout_len)

        df_holdout = self.df.iloc[holdout,]  # fmt: skip
        ser_holdout = self.ser.iloc[holdout]

        df = self.df.drop(index=self.df.index[holdout])
        ser = self.ser.drop(index=self.ser.index[holdout])

        self.idx += 1
        return df, df_holdout, ser, ser_holdout

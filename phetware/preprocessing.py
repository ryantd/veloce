from typing import List

import pandas as pd


class fillna(object):
    def __init__(self, col_selectors: List, value):
        self.col_selectors = col_selectors
        self.value = value
    
    def __call__(self, data):
        if isinstance(data, pd.DataFrame):
            return self._call_pandas_dataframe(data)
        else:
            raise NotImplementedError
    
    def _call_pandas_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        values = {col: self.value for col in self.col_selectors}
        return df.fillna(value=values)


class LabelEncoder(object):
    def __init__(self, col_selectors: List):
        self.col_selectors = col_selectors
    
    def __call__(self, data):
        if isinstance(data, pd.DataFrame):
            return self._call_pandas_dataframe(data)
        else:
            raise NotImplementedError

    def _call_pandas_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.col_selectors] = df[self.col_selectors].astype('category')
        for col in self.col_selectors:
            df[col] = df[col].cat.codes
        return df


class MinMaxScaler(object):
    def __init__(self, col_selector):
        self.col_selector = col_selector
    
    def __call__(self, data):
        if isinstance(data, pd.DataFrame):
            return self._call_pandas_dataframe(data)
        else:
            raise NotImplementedError

    def _call_pandas_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df_max = df[self.col_selector].max()
        df_min = df[self.col_selector].min()
        df[self.col_selector] = df[self.col_selector] / (df_max - df_min) + df_min
        return df
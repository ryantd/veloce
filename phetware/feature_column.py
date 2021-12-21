from typing import List

import pandas as pd
import pyarrow as pa

from phetware.inputs import SparseFeat, DenseFeat


class SparseFeatureColumn(object):
    def __init__(
        self,
        col_selectors: List,
        vocabulary_size_fn=lambda x, y: x[y].max() + 1,
        embedding_dim=4
    ):
        self.col_selectors = col_selectors
        self.embedding_dim = embedding_dim
        self.vocabulary_size_fn = vocabulary_size_fn

    def __call__(self, data):
        if isinstance(data, pd.DataFrame):
            return self._call_pandas_dataframe(data)
        else:
            raise NotImplementedError
    
    def _call_pandas_dataframe(self, df: pd.DataFrame) -> pa.Table:
        sparse_df = pd.DataFrame()
        for col in self.col_selectors:
            sparse_feat = SparseFeat(
                col,
                vocabulary_size=self.vocabulary_size_fn(df, col),
                embedding_dim=self.embedding_dim)
            sparse_df = sparse_df.append(
                sparse_feat._asdict(), ignore_index=True)
        return pd.DataFrame(sparse_df)


class DenseFeatureColumn(object):
    def __init__(self, col_selectors: List, dimension=1):
        self.col_selectors = col_selectors
        self.dimension = dimension
    
    def __call__(self, df):
        if isinstance(df, pd.DataFrame):
            return self._call_pandas_dataframe(df)
        else:
            raise NotImplementedError

    def _call_pandas_dataframe(self, _: pd.DataFrame) -> pd.DataFrame:
        dense_df = pd.DataFrame()
        for col in self.col_selectors:
            dense_feat = DenseFeat(col, self.dimension)
            dense_df = dense_df.append(dense_feat._asdict(), ignore_index=True)
        return pd.DataFrame(dense_df)

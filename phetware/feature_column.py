from typing import List

import pandas as pd

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
    
    def _call_pandas_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        sparse_df = pd.DataFrame()
        for col in self.col_selectors:
            sparse_feat = SparseFeat(
                col,
                vocabulary_size=self.vocabulary_size_fn(df, col),
                embedding_dim=self.embedding_dim)
            sparse_df = sparse_df.append(
                sparse_feat._asdict(), ignore_index=True)
        return sparse_df


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
        return dense_df


class FeatureColumnSet(object):
    def __init__(self, dnn_feature_columns, linear_feature_columns):
        self.dnn_fcs = dnn_feature_columns
        self.linear_fcs = linear_feature_columns
        self.all_fcs = dnn_feature_columns + linear_feature_columns
    
    def sorter(self):
        self.dnn_sparse_fcs, self.dnn_dence_fcs = [], []
        self.linear_sparse_fcs, self.linear_dence_fcs = [], []
        for x in self.dnn_fcs:
            if isinstance(x, SparseFeat): self.dnn_sparse_fcs.append(x)
            elif isinstance(x, DenseFeat): self.dnn_dence_fcs.append(x)
        for x in self.linear_fcs:
            if isinstance(x, SparseFeat): self.linear_sparse_fcs.append(x)
            elif isinstance(x, DenseFeat): self.linear_dence_fcs.append(x)

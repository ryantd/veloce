from typing import List

import pandas as pd

from phetware.inputs import SparseFeat, DenseFeat


class SparseFeatureDef(object):
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


class DenseFeatureDef(object):
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


class FeatureDefSet(object):
    def __init__(self, dnn_feature_defs, linear_feature_defs):
        self.dnn_defs = dnn_feature_defs
        self.linear_defs = linear_feature_defs
        self.all_defs = dnn_feature_defs + linear_feature_defs
    
    def sorter(self):
        self.dnn_sparse_defs, self.dnn_dence_defs = [], []
        self.linear_sparse_defs, self.linear_dence_defs = [], []
        for x in self.dnn_defs:
            if isinstance(x, SparseFeat): self.dnn_sparse_defs.append(x)
            elif isinstance(x, DenseFeat): self.dnn_dence_defs.append(x)
        for x in self.linear_defs:
            if isinstance(x, SparseFeat): self.linear_sparse_defs.append(x)
            elif isinstance(x, DenseFeat): self.linear_dence_defs.append(x)

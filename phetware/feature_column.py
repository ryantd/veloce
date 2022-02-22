from typing import List

import pandas as pd

from phetware.inputs import SparseFeat, DenseFeat


class SparseFeatureDef(object):
    def __init__(
        self,
        col_selectors: List,
        vocabulary_size_fn=lambda x, y: x[y].max() + 1,
        embedding_dim=4,
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
                column_idx=df.columns.get_loc(col) - 1,
                embedding_dim=self.embedding_dim,
            )
            sparse_df = sparse_df.append(sparse_feat._asdict(), ignore_index=True)
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

    def _call_pandas_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        dense_df = pd.DataFrame()
        for col in self.col_selectors:
            dense_feat = DenseFeat(
                col,
                column_idx=df.columns.get_loc(col) - 1,
                dimension=self.dimension,
            )
            dense_df = dense_df.append(dense_feat._asdict(), ignore_index=True)
        return dense_df


class FeatureDefSet(object):
    def __init__(self, def_mappings):
        self.all_defs = []
        self.def_types = []
        for long_key, v in def_mappings.items():
            k = f"{long_key.split('_feature_defs')[0]}_defs"
            setattr(self, k, v)
            self.all_defs += v
            self.def_types.append(k)

    def sorter(self):
        for t in self.def_types:
            for x in getattr(self, t):
                for feat_cls in [DenseFeat, SparseFeat]:
                    k = f"{t}_{feat_cls.key}"
                    if not hasattr(self, k):
                        setattr(self, k, [])
                    if isinstance(x, feat_cls):
                        prev_defs = getattr(self, k)
                        prev_defs.append(x)
                        setattr(self, k, prev_defs)

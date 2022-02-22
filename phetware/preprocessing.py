from typing import List

import torch
import pandas as pd
import ray
from pyarrow.csv import ConvertOptions

from phetware.feature_column import SparseFeatureDef, DenseFeatureDef


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
        df[self.col_selectors] = df[self.col_selectors].astype("category")
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


class DataLoader(object):
    def __init__(self, path, *, strings_can_be_null=True, rand_seed=2021):
        self.ds = ray.data.read_csv(
            path,
            convert_options=ConvertOptions(strings_can_be_null=strings_can_be_null),
        )
        self.rand_seed = rand_seed

    def set_sparse_features(
        self,
        feat_names,
        *,
        use_fillna=True,
        use_label_encoder=True,
        embedding_dim=4,
        default_fillna_value="-1",
        batch_format="pandas",
    ):
        self.sparse_feat_names = feat_names
        if use_fillna:
            self.ds = self.ds.map_batches(
                fillna(self.sparse_feat_names, default_fillna_value),
                batch_format=batch_format,
            )
        if use_label_encoder:
            self.ds = self.ds.map_batches(
                LabelEncoder(self.sparse_feat_names), batch_format=batch_format
            )
        sparse_defs_raw = self.ds.map_batches(
            SparseFeatureDef(
                self.sparse_feat_names,
                embedding_dim=embedding_dim,
            ),
            batch_format=batch_format,
        )
        self.sparse_defs = sparse_defs_raw.to_pandas().to_dict("records")
        return self

    def set_dense_features(
        self,
        feat_names,
        *,
        dim=1,
        use_fillna=True,
        use_minmax_scaler=True,
        default_fillna_value=0,
        batch_format="pandas",
    ):
        self.dense_feat_names = feat_names
        if use_fillna:
            self.ds = self.ds.map_batches(
                fillna(self.dense_feat_names, default_fillna_value),
                batch_format=batch_format,
            )
        if use_minmax_scaler:
            self.ds = self.ds.map_batches(
                MinMaxScaler(self.dense_feat_names), batch_format=batch_format
            )
        dense_defs_raw = self.ds.map_batches(
            DenseFeatureDef(self.dense_feat_names, dimension=dim),
            batch_format=batch_format,
        )
        self.dense_defs = dense_defs_raw.to_pandas().to_dict("records")
        return self

    def set_label_column(self, label_name):
        self.label_name = label_name
        return self

    def split(self, valid_split_factor=0.8, return_type="shard_dict"):
        valid_idx = int(self.ds.count() * valid_split_factor)
        train_dataset, validation_dataset = self.ds.random_shuffle(
            seed=self.rand_seed
        ).split_at_indices([valid_idx])
        self.train_dataset_pipeline = train_dataset.repeat().random_shuffle_each_window(
            seed=self.rand_seed
        )
        self.validation_dataset_pipeline = validation_dataset.repeat()
        if return_type == "tuple":
            return self.train_dataset_pipeline, self.validation_dataset_pipeline
        return {
            "train": self.train_dataset_pipeline,
            "validation": self.validation_dataset_pipeline,
        }

    def gen_torch_dataset_options(self, order):
        if not order or not all([o in {"dense", "sparse"} for o in order]):
            raise ValueError("Arg order should be given or invalid")
        feature_columns = getattr(self, f"{order[0]}_feat_names") + getattr(
            self, f"{order[1]}_feat_names"
        )
        return {
            "label_column": self.label_name,
            "feature_columns": feature_columns,
            "label_column_dtype": torch.float,
            "feature_column_dtypes": [torch.float] * len(feature_columns),
        }

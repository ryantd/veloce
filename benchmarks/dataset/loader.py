import os
from pathlib import Path

import torch
import ray
from pyarrow.csv import ConvertOptions

from phetware.preprocessing import fillna, MinMaxScaler, LabelEncoder
from phetware.feature_column import SparseFeatureDef, DenseFeatureDef
from .mapping import BUILTIN_DATASET_FEATS_MAPPING


def load_dataset_builtin(
    dataset_name="criteo_mini",
    feature_column_settings=None,
    valid_split_factor=0.8,
    test_split_factor=0.9,
    rand_seed=2021
):
    if dataset_name not in BUILTIN_DATASET_FEATS_MAPPING:
        raise ValueError("Dataset name is invalid")

    if not feature_column_settings:
        feature_column_settings = dict(
            dnn_dense=True, dnn_sparse=True,
            linear_dense=True, linear_sparse=False)

    # predefined feature names
    dense_features = BUILTIN_DATASET_FEATS_MAPPING[dataset_name]["dense"]
    sparse_features = BUILTIN_DATASET_FEATS_MAPPING[dataset_name]["sparse"]
    label_column = BUILTIN_DATASET_FEATS_MAPPING[dataset_name]["label"]
    path = BUILTIN_DATASET_FEATS_MAPPING[dataset_name]["path"]

    # preprocess dataset
    ds = ray.data \
        .read_csv(
            path, convert_options=ConvertOptions(strings_can_be_null=True)) \
        .map_batches(fillna(sparse_features, "-1"), batch_format="pandas") \
        .map_batches(fillna(dense_features, 0), batch_format="pandas") \
        .map_batches(LabelEncoder(sparse_features), batch_format="pandas") \
        .map_batches(MinMaxScaler(dense_features), batch_format="pandas")

    # process feature columns
    sparse_defs = ds.map_batches(
        SparseFeatureDef(sparse_features), batch_format="pandas")
    dense_defs = ds.map_batches(
        DenseFeatureDef(dense_features), batch_format="pandas")
    dnn_feature_defs, linear_feature_defs = select_feature_defs(
        feature_column_settings,
        dense_defs=dense_defs.to_pandas().to_dict('records'),
        sparse_defs=sparse_defs.to_pandas().to_dict('records'))

    # split dataset
    valid_idx = int(ds.count() * valid_split_factor)
    test_idx = int(ds.count() * test_split_factor)
    train_dataset, validation_dataset, test_dataset = \
        ds.random_shuffle(seed=rand_seed).split_at_indices([valid_idx, test_idx])
    train_dataset_pipeline = \
        train_dataset.repeat().random_shuffle_each_window(seed=rand_seed)
    validation_dataset_pipeline = validation_dataset.repeat()
    test_dataset_pipeline = test_dataset.repeat()

    # datasets and defs generating
    datasets = {
        "train": train_dataset_pipeline,
        "validation": validation_dataset_pipeline,
        "test": test_dataset_pipeline
    }
    feature_defs = {
        "dnn": dnn_feature_defs,
        "linear": linear_feature_defs
    }
    torch_dataset_options = {
        "label_column": label_column,
        "feature_columns": dense_features + sparse_features,
        "label_column_dtype": torch.float,
        "feature_column_dtypes": [torch.float] * len(dense_features + sparse_features)
    }
    return datasets, feature_defs, torch_dataset_options


def select_feature_defs(bool_settings, dense_defs, sparse_defs):
    settings = {k: int(v) for k, v in bool_settings.items()}
    def combine(t):
        return dense_defs * settings[f"{t}_dense"] + sparse_defs * settings[f"{t}_sparse"]
    return combine("dnn"), combine("linear")

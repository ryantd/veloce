import torch
import ray
from pyarrow.csv import ConvertOptions

from phetware.preprocessing import fillna, MinMaxScaler, LabelEncoder
from phetware.feature_column import SparseFeatureDef, DenseFeatureDef
from .mapping import BUILTIN_DATASET_FEATS_MAPPING


def load_dataset_builtin(
    dataset_name="criteo_mini",
    feature_def_settings=None,
    valid_split_factor=0.8,
    test_split_factor=0.9,
    rand_seed=2021
):
    if dataset_name not in BUILTIN_DATASET_FEATS_MAPPING:
        raise ValueError("Dataset name is invalid")

    if not feature_def_settings:
        raise ValueError("Arg feature_column_settings should be given")

    # predefined feature names
    dense_feat_names = BUILTIN_DATASET_FEATS_MAPPING[dataset_name]["dense"]
    sparse_feat_names = BUILTIN_DATASET_FEATS_MAPPING[dataset_name]["sparse"]
    label_name = BUILTIN_DATASET_FEATS_MAPPING[dataset_name]["label"]
    path = BUILTIN_DATASET_FEATS_MAPPING[dataset_name]["path"]

    # preprocess dataset
    ds = ray.data \
        .read_csv(
            path, convert_options=ConvertOptions(strings_can_be_null=True)) \
        .map_batches(fillna(sparse_feat_names, "-1"), batch_format="pandas") \
        .map_batches(fillna(dense_feat_names, 0), batch_format="pandas") \
        .map_batches(LabelEncoder(sparse_feat_names), batch_format="pandas") \
        .map_batches(MinMaxScaler(dense_feat_names), batch_format="pandas")

    # process feature defs
    sparse_defs = ds.map_batches(
        SparseFeatureDef(sparse_feat_names), batch_format="pandas")
    dense_defs = ds.map_batches(
        DenseFeatureDef(dense_feat_names), batch_format="pandas")
    dense_defs = dense_defs.to_pandas().to_dict('records')
    sparse_defs = sparse_defs.to_pandas().to_dict('records')

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
        k: dense_defs * int(v["dense"]) + sparse_defs * int(v["sparse"])
        for k, v in feature_def_settings.items()}
    torch_dataset_options = {
        "label_column": label_name,
        "feature_columns": dense_feat_names + sparse_feat_names,
        "label_column_dtype": torch.float,
        "feature_column_dtypes": [torch.float] * len(dense_feat_names + sparse_feat_names)
    }
    return datasets, feature_defs, torch_dataset_options

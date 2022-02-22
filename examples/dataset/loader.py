import torch
import ray
from pyarrow.csv import ConvertOptions

from phetware.preprocessing import fillna, MinMaxScaler, LabelEncoder
from phetware.feature_column import SparseFeatureDef, DenseFeatureDef
from .mapping import BUILTIN_DATASET_FEATS_MAPPING

FEAT_DEF_RESERVED_GLOBAL = "_global_settings"
EMBEDDING_DIM_DEFAULT = 4


def load_dataset(
    dataset_name="criteo_mini",
    feature_def_settings=None,
    valid_split_factor=0.8,
    rand_seed=2021,
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
    ds = (
        ray.data.read_csv(
            path, convert_options=ConvertOptions(strings_can_be_null=True)
        )
        .map_batches(fillna(sparse_feat_names, "-1"), batch_format="pandas")
        .map_batches(fillna(dense_feat_names, 0), batch_format="pandas")
        .map_batches(LabelEncoder(sparse_feat_names), batch_format="pandas")
        .map_batches(MinMaxScaler(dense_feat_names), batch_format="pandas")
    )

    # process feature defs
    try:
        embedding_dim = feature_def_settings[FEAT_DEF_RESERVED_GLOBAL]["embedding_dim"]
    except:
        embedding_dim = EMBEDDING_DIM_DEFAULT
    sparse_defs = ds.map_batches(
        SparseFeatureDef(
            sparse_feat_names,
            embedding_dim=embedding_dim,
        ),
        batch_format="pandas",
    )
    dense_defs = ds.map_batches(
        DenseFeatureDef(dense_feat_names), batch_format="pandas"
    )
    dense_defs = dense_defs.to_pandas().to_dict("records")
    sparse_defs = sparse_defs.to_pandas().to_dict("records")

    # split dataset
    valid_idx = int(ds.count() * valid_split_factor)
    train_dataset, validation_dataset = ds.random_shuffle(
        seed=rand_seed
    ).split_at_indices([valid_idx])
    train_dataset_pipeline = train_dataset.repeat().random_shuffle_each_window(
        seed=rand_seed
    )
    validation_dataset_pipeline = validation_dataset.repeat()

    # datasets and defs generating
    datasets = {
        "train": train_dataset_pipeline,
        "validation": validation_dataset_pipeline,
    }
    feature_defs = {
        k: dense_defs * int(v["dense"]) + sparse_defs * int(v["sparse"])
        for k, v in feature_def_settings.items()
        if k != FEAT_DEF_RESERVED_GLOBAL
    }
    torch_dataset_options = {
        "label_column": label_name,
        "feature_columns": dense_feat_names + sparse_feat_names,
        "label_column_dtype": torch.float,
        "feature_column_dtypes": [torch.float]
        * len(dense_feat_names + sparse_feat_names),
    }
    return datasets, feature_defs, torch_dataset_options

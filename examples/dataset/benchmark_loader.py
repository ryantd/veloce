from phetware.preprocessing import DataLoader

BENCHMARK_DATASET_DEFAULT = "examples/dataset/ctr/criteo_100k.txt"


def load_benchmark_dataset(
    data_path=BENCHMARK_DATASET_DEFAULT,
    *,
    feature_def_settings=None,
    valid_split_factor=0.85,
    rand_seed=2021,
    embedding_dim=4,
    separate_valid_dataset=True,
):
    if not feature_def_settings:
        raise ValueError("Arg feature_column_settings should be given")

    dataloader = DataLoader(data_path, rand_seed=rand_seed)
    dataloader = (
        dataloader.set_label_column(label_name="label")
        .set_dense_features(feat_names=[f"I{i}" for i in range(1, 14)])
        .set_sparse_features(
            feat_names=[f"C{i}" for i in range(1, 27)], embedding_dim=embedding_dim
        )
    )
    datasets = dataloader.split(valid_split_factor=valid_split_factor)
    dense_defs = dataloader.dense_defs
    sparse_defs = dataloader.sparse_defs
    torch_dataset_options = dataloader.gen_torch_dataset_options(
        order=("dense", "sparse")
    )
    feature_defs = {
        k: dense_defs * int(v["dense"]) + sparse_defs * int(v["sparse"])
        for k, v in feature_def_settings.items()
    }
    if separate_valid_dataset:
        validation_ds = datasets.pop("validation")
        return (datasets, validation_ds), feature_defs, torch_dataset_options
    return datasets, feature_defs, torch_dataset_options

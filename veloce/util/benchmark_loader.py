from veloce.preprocessing import DataLoader

BENCHMARK_DATASET_DEFAULT = "examples/dataset/ctr/criteo_mini.txt"


def load_benchmark_dataset(
    data_path=BENCHMARK_DATASET_DEFAULT,
    *,
    valid_split_factor=0.85,
    rand_seed=2021,
    embedding_dim=4,
    separate_valid_dataset=True,
):
    dataloader = DataLoader(data_path, rand_seed=rand_seed)
    dataloader = (
        dataloader.set_label_column(label_name="label")
        .set_dense_features(feat_names=[f"I{i}" for i in range(1, 14)])
        .set_sparse_features(
            feat_names=[f"C{i}" for i in range(1, 27)], embedding_dim=embedding_dim
        )
        .set_features_order(order=("dense", "sparse"))
    )
    datasets = dataloader.split(valid_split_factor=valid_split_factor)
    torch_dataset_options = dataloader.gen_torch_dataset_options()
    feature_defs = {"dense": dataloader.dense_defs, "sparse": dataloader.sparse_defs}
    if separate_valid_dataset:
        validation_ds = datasets.pop("validation")
        return (datasets, validation_ds), feature_defs, torch_dataset_options
    return datasets, feature_defs, torch_dataset_options

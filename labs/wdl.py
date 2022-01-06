import torch
import torch.nn as nn
import torchmetrics
import ray
from ray.train import Trainer
from ray.train.callbacks import JsonLoggerCallback, TBXLoggerCallback
from pyarrow.csv import ConvertOptions
from sklearn.metrics import log_loss

from phetware.train_fn import WideAndDeep
from phetware.preprocessing import fillna, MinMaxScaler, LabelEncoder
from phetware.feature_column import SparseFeatureColumn, DenseFeatureColumn
from phetware.optimizer import OptimizerStack, FTRL
from phetware.loss_fn import LossFnStack

sparse_features = [f'C{i}' for i in range(1, 27)]
dense_features = [f'I{i}' for i in range(1, 14)]
VALID_SPLIT_FACTOR = 0.8
TEST_SPLIT_FACTOR = 0.9
RAND_SEED = 2021


def get_dataset_and_fc():
    # preprocess dataset
    ds = ray.data \
        .read_csv(
            'labs/data/criteo_sample.txt',
            convert_options=ConvertOptions(strings_can_be_null=True)) \
        .map_batches(fillna(sparse_features, "-1"), batch_format="pandas") \
        .map_batches(fillna(dense_features, 0), batch_format="pandas") \
        .map_batches(LabelEncoder(sparse_features), batch_format="pandas") \
        .map_batches(MinMaxScaler(dense_features), batch_format="pandas")

    # process feature columns
    sparse_fc = ds.map_batches(
        SparseFeatureColumn(sparse_features), batch_format="pandas")
    dense_fc = ds.map_batches(
        DenseFeatureColumn(dense_features), batch_format="pandas")
    fixlen_fc = sparse_fc.to_pandas().to_dict('records') + dense_fc.to_pandas().to_dict('records')
    dnn_feature_columns = fixlen_fc
    linear_feature_columns = dense_fc.to_pandas().to_dict('records')

    # split dataset
    valid_idx = int(ds.count() * VALID_SPLIT_FACTOR)
    test_idx = int(ds.count() * TEST_SPLIT_FACTOR)
    train_dataset, validation_dataset, test_dataset = \
        ds.random_shuffle(seed=RAND_SEED).split_at_indices([valid_idx, test_idx])
    train_dataset_pipeline = \
        train_dataset.repeat().random_shuffle_each_window(seed=RAND_SEED)
    validation_dataset_pipeline = validation_dataset.repeat()
    test_dataset_pipeline = test_dataset.repeat()

    # datasets and fcs generating
    datasets = {
        "train": train_dataset_pipeline,
        "validation": validation_dataset_pipeline,
        "test": test_dataset_pipeline
    }
    feature_columns = {
        "dnn": dnn_feature_columns,
        "linear": linear_feature_columns
    }
    return datasets, feature_columns


def train_wdl_dist(num_workers=2, use_gpu=False):
    datasets, feature_columns = get_dataset_and_fc()

    trainer = Trainer("torch", num_workers=num_workers, use_gpu=use_gpu)
    trainer.start()
    
    results = trainer.run(
        train_func=WideAndDeep(),
        dataset=datasets,
        callbacks=[JsonLoggerCallback(), TBXLoggerCallback()],
        # support fault tolerance with checkpoints
        # checkpoint=dict(),
        config={
            "dnn_feature_columns": feature_columns["dnn"],
            "linear_feature_columns": feature_columns["linear"],
            "epochs": 10,
            "batch_size": 256,
            "dnn_dropout": 0.2,
            "seed": RAND_SEED,
            # support multiple loss functions with fixed weight
            "loss_fn": LossFnStack(
                dict(fn=nn.BCELoss(), weight=0.2),
                dict(fn=nn.HingeEmbeddingLoss(), weight=0.8)),
            # support multiple optimizers
            "optimizer": OptimizerStack(
                dict(cls=torch.optim.Adagrad, model_key="deep_model"),
                dict(cls=FTRL, args=dict(alpha=1.0, beta=1.0, l1=1.0, l2=1.0), model_key="wide_model")),
            # support torchmetrics and sklearn metric funcs
            "metric_fns": [torchmetrics.AUROC(), log_loss],
            "output_fn": torch.softmax,
            "output_fn_args": dict(dim=0),
            "torch_dataset_options": dict(
                label_column="label",
                feature_columns=sparse_features + dense_features,
                label_column_dtype=torch.float,
                feature_column_dtypes=[torch.float] * (len(sparse_features) + len(dense_features)))
        })
    trainer.shutdown()
    print(f"Results: {results[0][-1]}") # AUROC: 0.5934, log_loss: 0.9292


if __name__ == "__main__":
    ray.init(num_cpus=1 + 2)
    train_wdl_dist()

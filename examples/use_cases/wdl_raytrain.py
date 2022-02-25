import torch
import torch.nn as nn
from torchmetrics.functional import auroc
import ray
from ray.train import Trainer
from ray.train.callbacks import JsonLoggerCallback, TBXLoggerCallback
from sklearn.metrics import log_loss

from phetware.train_fn import WideAndDeep
from phetware.optimizer import OptimizerStack, FTRL
from phetware.loss_fn import LossFnStack
from phetware.util import pprint_results
from phetware.preprocessing import DataLoader


def train_wdl_dist(num_workers=2, use_gpu=False, rand_seed=2021):
    dataloader = DataLoader("examples/dataset/ctr/criteo_100k.txt")
    dataloader = (
        dataloader.set_label_column(label_name="label")
        .set_dense_features(feat_names=[f"I{i}" for i in range(1, 14)])
        .set_sparse_features(
            feat_names=[f"C{i}" for i in range(1, 27)], embedding_dim=1
        )
        # this order should follow the data file
        .set_features_order(order=("dense", "sparse"))
    )
    datasets = dataloader.split()
    dense_defs = dataloader.dense_defs
    sparse_defs = dataloader.sparse_defs
    torch_dataset_options = dataloader.gen_torch_dataset_options()

    # dense_defs is like,
    # [{'name': 'I1', 'dimension': 1.0, 'dtype': 'float32', 'column_idx': 1.0,
    #   'feat_type': 'DenseFeat'},
    #  {'name': 'I2', 'dimension': 1.0, 'dtype': 'float32', 'column_idx': 2.0,
    #   'feat_type': 'DenseFeat'}, ...]

    # sparse_defs is like,
    # [{'name': 'C1', 'vocabulary_size': 557.0, 'embedding_dim': 1.0,
    #   'dtype': 'int32', 'group_name': 'default_group', 'column_idx': 14.0,
    #   'feat_type': 'SparseFeat'},
    #  {'name': 'C2', 'vocabulary_size': 507.0, 'embedding_dim': 1.0,
    #   'dtype': 'int32', 'group_name': 'default_group', 'column_idx': 15.0,
    #   'feat_type': 'SparseFeat'}, ...]

    trainer = Trainer("torch", num_workers=num_workers, use_gpu=use_gpu)
    trainer.start()

    results = trainer.run(
        train_func=WideAndDeep,
        dataset=datasets,
        callbacks=[JsonLoggerCallback(), TBXLoggerCallback()],
        # support fault tolerance with checkpoints
        # checkpoint=dict(),
        config={
            # API: defined in WideAndDeep model
            "dense_feature_defs": dense_defs,
            "sparse_feature_defs": sparse_defs,
            "output_fn": torch.softmax,
            "output_fn_args": dict(dim=0),
            "seed": rand_seed,
            "dnn_dropout": 0.5,
            # API: defined in train lifecycle
            "epochs": 50,
            "batch_size": 32,
            "loss_fn": LossFnStack(
                # support multiple loss functions with fixed weight
                dict(fn=nn.BCELoss(), weight=0.8),
                dict(fn=nn.HingeEmbeddingLoss(), weight=0.2),
            ),
            "optimizer": OptimizerStack(
                # support multiple optimizers
                dict(cls=torch.optim.Adam, model_key="deep_model"),
                dict(
                    cls=FTRL,
                    args=dict(lr=0.925, weight_decay=1e-3),
                    model_key="wide_model",
                ),
            ),
            "metric_fns": [
                # support torchmetrics and sklearn metric funcs
                auroc,
                log_loss,
            ],
            "dataset_options": torch_dataset_options,
        },
    )
    trainer.shutdown()
    pprint_results(results, print_interval=10)


if __name__ == "__main__":
    ray.init(num_cpus=1 + 2)
    train_wdl_dist()

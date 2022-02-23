import torch
import torch.nn as nn
import torchmetrics
import ray
from ray.train import Trainer
from ray.train.callbacks import JsonLoggerCallback, TBXLoggerCallback
from sklearn.metrics import log_loss

from phetware.train_fn import WideAndDeep
from phetware.optimizer import OptimizerStack, FTRL
from phetware.loss_fn import LossFnStack
from phetware.util import pprint_results
from examples.dataset import load_dataset


def train_wdl_dist(num_workers=2, use_gpu=False, rand_seed=2021):
    datasets, feature_defs, torch_dataset_options = load_dataset(
        dataset_name="criteo_mini",
        feature_def_settings={
            "dnn": {"dense": True, "sparse": True},
            "linear": {"dense": True, "sparse": False},
        },
    )

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
            "dnn_feature_defs": feature_defs["dnn"],
            "linear_feature_defs": feature_defs["linear"],
            "seed": rand_seed,
            "output_fn": torch.softmax,
            "output_fn_args": dict(dim=0),
            "dnn_dropout": 0.2,
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
                    args=dict(lr=4.25, weight_decay=1e-3),
                    model_key="wide_model",
                ),
            ),
            "metric_fns": [
                # support torchmetrics and sklearn metric funcs
                torchmetrics.AUROC(),
                log_loss,
                torchmetrics.MeanSquaredError(squared=False),
            ],
            "torch_dataset_options": torch_dataset_options,
        },
    )
    trainer.shutdown()
    pprint_results(results, print_interval=10)


if __name__ == "__main__":
    ray.init(num_cpus=1 + 2)
    train_wdl_dist()
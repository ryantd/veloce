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
from benchmarks.dataset import load_dataset_builtin


def train_wdl_dist(num_workers=2, use_gpu=False, rand_seed=2021):
    datasets, feature_defs, torch_dataset_options = load_dataset_builtin("criteo_mini")

    trainer = Trainer("torch", num_workers=num_workers, use_gpu=use_gpu)
    trainer.start()

    results = trainer.run(
        train_func=WideAndDeep(),
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
            "epochs": 10,
            "batch_size": 256,
            "loss_fn": LossFnStack(
                # support multiple loss functions with fixed weight
                dict(fn=nn.BCELoss(), weight=0.2),
                dict(fn=nn.HingeEmbeddingLoss(), weight=0.8)),
            "optimizer": OptimizerStack(
                # support multiple optimizers
                dict(cls=torch.optim.Adagrad, model_key="deep_model"),
                dict(cls=FTRL, args=dict(alpha=1.0, beta=1.0, l1=1.0, l2=1.0), model_key="wide_model")),
            "metric_fns": [
                # support torchmetrics and sklearn metric funcs
                torchmetrics.AUROC(), log_loss],
            "torch_dataset_options": torch_dataset_options
        })
    trainer.shutdown()
    print(f"Results: {results[0][-1]}") # AUROC: 0.5934, log_loss: 0.9292


if __name__ == "__main__":
    ray.init(num_cpus=1 + 2)
    train_wdl_dist()

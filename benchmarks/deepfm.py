import torch
import torch.nn as nn
import torchmetrics
import ray
from ray.train import Trainer
from ray.train.callbacks import JsonLoggerCallback, TBXLoggerCallback
from sklearn.metrics import log_loss

from phetware.train_fn import DeepFM
from benchmarks.dataset import load_dataset_builtin


def train_deepfm_dist(num_workers=2, use_gpu=False, rand_seed=2021):
    datasets, feature_defs, torch_dataset_options = load_dataset_builtin(
        dataset_name="criteo_mini",
        feature_def_settings={
            "fm_1": {"dense": True, "sparse": True},
            "fm_2": {"dense": False, "sparse": True},
            "dnn": {"dense": True, "sparse": True}})

    trainer = Trainer("torch", num_workers=num_workers, use_gpu=use_gpu)
    trainer.start()

    results = trainer.run(
        train_func=DeepFM(),
        dataset=datasets,
        callbacks=[JsonLoggerCallback(), TBXLoggerCallback()],
        config={
            "fm_1_feature_defs": feature_defs["fm_1"],
            "fm_2_feature_defs": feature_defs["fm_2"],
            "dnn_feature_defs": feature_defs["dnn"],
            "seed": rand_seed,
            "output_fn": torch.softmax,
            "output_fn_args": dict(dim=0),
            "dnn_dropout": 0.2,
            "epochs": 10,
            "batch_size": 256,
            "loss_fn": nn.BCELoss(),
            "optimizer": torch.optim.Adagrad,
            "metric_fns": [torchmetrics.AUROC(), log_loss],
            "torch_dataset_options": torch_dataset_options,
            "summary_nn_arch": True
        })
    trainer.shutdown()
    print(f"Results: {results[0][-1]}") # AUROC: 0.5934, log_loss: 0.9292


if __name__ == "__main__":
    ray.init(num_cpus=1 + 2)
    train_deepfm_dist()

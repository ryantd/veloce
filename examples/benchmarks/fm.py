import torch
import torch.nn as nn
from torchmetrics.functional import auroc

from phetware.util import pprint_results
from phetware.environ import environ_validate
from phetware.model.torch import FM
from phetware import NeuralNetTrainer
from examples.dataset import load_benchmark_dataset


def train_fm_dist(num_workers=2, use_gpu=False, rand_seed=2021):
    datasets, feature_defs, torch_dataset_options = load_benchmark_dataset()
    train_ds, valid_ds = datasets

    trainer = NeuralNetTrainer(
        # module and dataset configs
        module=FM,
        module_params={
            "dense_feature_defs": feature_defs["dense"],
            "sparse_feature_defs": feature_defs["sparse"],
            "seed": rand_seed,
        },
        dataset=train_ds,
        dataset_options=torch_dataset_options,
        shared_validation_dataset=valid_ds,
        # trainer configs
        epochs=50,
        batch_size=512,
        loss_fn=nn.BCELoss(),
        optimizer=torch.optim.Adam,
        optimizer_args={
            "lr": 1e-4,
            "weight_decay": 1e-3,
        },
        metric_fns=[auroc],
        use_early_stopping=True,
        early_stopping_args={"patience": 2},
        num_workers=num_workers,
        use_gpu=use_gpu,
        callbacks=["json", "tbx"],
    )
    results = trainer.run()
    pprint_results(results)
    """
    epochs 50
    valid/BCELoss avg: 0.49830	valid/auroc avg: 0.74110
    """


if __name__ == "__main__":
    environ_validate(n_cpus=1 + 2)
    train_fm_dist(num_workers=2)

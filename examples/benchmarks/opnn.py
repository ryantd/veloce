import torch
import torch.nn as nn
from torchmetrics.functional import auroc

from enscale.model.ctr import PNN
from enscale.util import pprint_results
from enscale.environ import environ_validate
from enscale import NeuralNetTrainer
from enscale.util import load_benchmark_dataset


def train_opnn_dist(num_workers=2, use_gpu=False, rand_seed=2021):
    datasets, feature_defs, torch_dataset_options = load_benchmark_dataset()
    train_ds, valid_ds = datasets

    trainer = NeuralNetTrainer(
        # module and dataset configs
        module=PNN,
        module_params={
            "dense_feature_defs": feature_defs["dense"],
            "sparse_feature_defs": feature_defs["sparse"],
            "use_outer": True,
            "use_inner": False,
            "dnn_dropout": 0.5,
            "dnn_hidden_units": (200, 200, 200),
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
    epochs 20 ES
    valid/BCELoss avg: 0.49522	valid/auroc avg: 0.74774
    """


if __name__ == "__main__":
    environ_validate(n_cpus=1 + 2)
    train_opnn_dist(num_workers=2)

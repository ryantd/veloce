import torch
import torch.nn as nn
from torchmetrics.functional import auroc

from phetware.model.torch import FNN
from phetware.util import pprint_results
from phetware.environ import environ_validate
from phetware import NeuralNetTrainer
from benchmarks.dataset import load_dataset_builtin


def train_fnn_dist(num_workers=2, use_gpu=False, rand_seed=2021):
    datasets, feature_defs, torch_dataset_options = load_dataset_builtin(
        dataset_name="criteo_10k",
        feature_def_settings={
            "dnn": {"dense": True, "sparse": True},
        },
    )

    trainer = NeuralNetTrainer(
        # module and dataset configs
        module=FNN,
        module_params={
            "dnn_feature_defs": feature_defs["dnn"],
            "pre_trained_mode": True,
            "seed": rand_seed,
            "dnn_dropout": 0.5,
        },
        dataset=datasets,
        dataset_options=torch_dataset_options,
        # trainer configs
        batch_size=512,
        loss_fn=nn.BCELoss(),
        optimizer=torch.optim.Adam,
        optimizer_args={
            "weight_decay": 1e-3,
        },
        metric_fns=[auroc],
        use_static_graph=True,
        num_workers=num_workers,
        use_gpu=use_gpu,
        callbacks=["json", "tbx"],
    )
    results = trainer.run(
        multi_runs=[
            {"epochs": 10},
            {"epochs": 10, "module_params": {"pre_trained_mode": False}}
        ]
    )
    pprint_results(results)
    """
    optimizer=Adam
    valid/BCELoss: 0.49254	valid/auroc: 0.75292
    """


if __name__ == "__main__":
    environ_validate(num_cpus=1 + 2)
    train_fnn_dist(num_workers=2)

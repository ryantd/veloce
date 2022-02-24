import torch
import torch.nn as nn
from torchmetrics.functional import auroc

from phetware.model.torch import WideAndDeep
from phetware.util import pprint_results
from phetware import NeuralNetTrainer
from phetware.environ import environ_validate
from examples.dataset import load_benchmark_dataset


def train_wdl_dist(num_workers=2, use_gpu=False, rand_seed=2021):
    datasets, feature_defs, torch_dataset_options = load_benchmark_dataset(
        feature_def_settings={
            "dnn": {"dense": True, "sparse": True},
            "fm": {"dense": True, "sparse": False},
        },
    )
    train_ds, valid_ds = datasets

    trainer = NeuralNetTrainer(
        # module and dataset configs
        module=WideAndDeep,
        module_params={
            "dnn_feature_defs": feature_defs["dnn"],
            "linear_feature_defs": feature_defs["fm"],
            "dnn_dropout": 0.5,
            "dnn_hidden_units": (200, 200, 200),
            "use_fm": True,
            "seed": rand_seed,
        },
        dataset=train_ds,
        dataset_options=torch_dataset_options,
        shared_validation_dataset=valid_ds,
        # trainer configs
        epochs=20,
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
    valid/BCELoss avg: 0.48819	valid/auroc avg: 0.75234
    """


if __name__ == "__main__":
    environ_validate(num_cpus=1 + 2)
    train_wdl_dist(num_workers=2)

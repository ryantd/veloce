import torch
import torch.nn as nn
from torchmetrics.functional import auroc

from phetware.util import pprint_results
from phetware.environ import environ_validate
from phetware.model.torch import FM
from phetware import NeuralNetTrainer
from examples.dataset import load_benchmark_dataset


def train_fm_dist(num_workers=2, use_gpu=False, rand_seed=2021):
    datasets, feature_defs, torch_dataset_options = load_benchmark_dataset(
        feature_def_settings={
            "fm": {"dense": True, "sparse": True},
        },
    )

    trainer = NeuralNetTrainer(
        # module and dataset configs
        module=FM,
        module_params={
            "fm_feature_defs": feature_defs["fm"],
            "seed": rand_seed,
        },
        dataset=datasets,
        dataset_options=torch_dataset_options,
        # trainer configs
        epochs=20,
        batch_size=512,
        loss_fn=nn.BCELoss(),
        optimizer=torch.optim.Adam,
        metric_fns=[auroc],
        num_workers=num_workers,
        use_gpu=use_gpu,
        callbacks=["json", "tbx"],
    )
    results = trainer.run()
    pprint_results(results)
    """
    optimizer=Adam
    valid/BCELoss: 0.50188	valid/auroc: 0.74113
    """


if __name__ == "__main__":
    environ_validate(num_cpus=1 + 2)
    train_fm_dist(num_workers=2)

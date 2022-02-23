import torch.nn as nn
from torchmetrics.functional import auroc

from phetware.util import pprint_results
from phetware.environ import environ_validate
from phetware.optimizer import FTRL
from phetware.model.torch import LR
from phetware import NeuralNetTrainer
from examples.dataset import load_benchmark_dataset


def train_lr_dist(num_workers=2, use_gpu=False, rand_seed=2021):
    datasets, feature_defs, torch_dataset_options = load_benchmark_dataset(
        feature_def_settings={
            "linear": {"dense": True, "sparse": True},
        },
    )

    trainer = NeuralNetTrainer(
        # module and dataset configs
        module=LR,
        module_params={
            "linear_feature_defs": feature_defs["linear"],
            "seed": rand_seed,
        },
        dataset=datasets,
        dataset_options=torch_dataset_options,
        # trainer configs
        epochs=20,
        batch_size=512,
        loss_fn=nn.BCELoss(),
        optimizer=FTRL,
        metric_fns=[auroc],
        num_workers=num_workers,
        use_gpu=use_gpu,
        callbacks=["json", "tbx"],
    )
    results = trainer.run()
    pprint_results(results)
    """
    optimizer=FTRL
    valid/BCELoss avg: 0.52129	valid/auroc avg: 0.70726
    """


if __name__ == "__main__":
    environ_validate(num_cpus=1 + 2)
    train_lr_dist(num_workers=2)

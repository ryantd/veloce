import torch
import torch.nn as nn
from torchmetrics.functional import auroc

from phetware.model.torch import PNN
from phetware.util import pprint_results
from phetware.environ import environ_validate
from phetware import NeuralNetTrainer
from examples.dataset import load_benchmark_dataset


def train_opnn_dist(num_workers=2, use_gpu=False, rand_seed=2021):
    datasets, feature_defs, torch_dataset_options = load_benchmark_dataset(
        feature_def_settings={"dnn": {"dense": True, "sparse": True}},
    )

    trainer = NeuralNetTrainer(
        # module and dataset configs
        module=PNN,
        module_params={
            "dnn_feature_defs": feature_defs["dnn"],
            "use_outter": True,
            "use_inner": False,
            "dnn_dropout": 0.5,
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
        use_early_stopping=True,
        early_stopping_args={"patience": 2},
        num_workers=num_workers,
        use_gpu=use_gpu,
        callbacks=["json", "tbx"],
    )
    results = trainer.run()
    pprint_results(results)
    """
    optimizer=Adam
    early_stopping patience=2
    valid/BCELoss: 0.49203	valid/auroc: 0.75427
    """


if __name__ == "__main__":
    environ_validate(num_cpus=1 + 2)
    train_opnn_dist(num_workers=2)

import torch
import torch.nn as nn
from torchmetrics.functional import auroc

from phetware.model.torch import WideAndDeep
from phetware.optimizer import OptimizerStack, FTRL
from phetware.util import pprint_results
from phetware import NeuralNetTrainer
from phetware.environ import environ_validate
from examples.dataset import load_benchmark_dataset


def train_wdl_dist(num_workers=2, use_gpu=False, rand_seed=2021):
    datasets, feature_defs, torch_dataset_options = load_benchmark_dataset(
        feature_def_settings={
            "dnn": {"dense": True, "sparse": True},
            "linear": {"dense": True, "sparse": True},
        },
    )

    trainer = NeuralNetTrainer(
        # module and dataset configs
        module=WideAndDeep,
        module_params={
            "dnn_feature_defs": feature_defs["dnn"],
            "linear_feature_defs": feature_defs["linear"],
            "seed": rand_seed,
            "output_fn": torch.sigmoid,
            "dnn_dropout": 0.5,
        },
        dataset=datasets,
        dataset_options=torch_dataset_options,
        # trainer configs
        epochs=20,
        batch_size=512,
        loss_fn=nn.BCELoss(),
        optimizer=OptimizerStack(
            dict(
                cls=torch.optim.Adam,
                args=dict(weight_decay=1e-3),
                model_key="deep_model",
            ),
            dict(
                cls=FTRL, args=dict(lr=4.25, weight_decay=1e-3), model_key="wide_model"
            ),
        ),
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
    weight_decay=1e-3
    valid/BCELoss: 0.49301	valid/auroc: 0.75244

    optimizer=FTRL + Adam
    early_stopping patience=2
    weight_decay=1e-3
    FTRL lr=4.25
    valid/BCELoss: 0.50941	valid/auroc: 0.73046
    """


if __name__ == "__main__":
    environ_validate(num_cpus=1 + 2)
    train_wdl_dist(num_workers=2)

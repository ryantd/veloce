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
    datasets, feature_defs, torch_dataset_options = load_benchmark_dataset()
    train_ds, valid_ds = datasets

    trainer = NeuralNetTrainer(
        # module and dataset configs
        module=WideAndDeep,
        module_params={
            "dense_feature_defs": feature_defs["dense"],
            "sparse_feature_defs": feature_defs["sparse"],
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
        optimizer=OptimizerStack(
            dict(
                cls=torch.optim.Adam,
                args=dict(weight_decay=1e-3, lr=1e-4),
                model_key="deep_model",
            ),
            dict(
                cls=FTRL, args=dict(lr=0.925, weight_decay=1e-3), model_key="wide_model"
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
    epochs 36 ES
    valid/BCELoss avg: 0.50481	valid/auroc avg: 0.73152
    """


if __name__ == "__main__":
    environ_validate(num_cpus=1 + 2)
    train_wdl_dist(num_workers=2)

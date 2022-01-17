import torch
import torch.nn as nn
import torchmetrics
from sklearn.metrics import log_loss

from phetware.model.torch import WideAndDeep
from phetware.util import pprint_results
from phetware import NeuralNetTrainer
from phetware.optimizer import OptimizerStack, FTRL
from phetware.loss_fn import LossFnStack
from phetware.environ import environ_validate
from benchmarks.dataset import load_dataset_builtin


def train_wdl_dist(num_workers=2, use_gpu=False, rand_seed=2021):
    datasets, feature_defs, torch_dataset_options = load_dataset_builtin(
        dataset_name="criteo_mini",
        feature_def_settings={
            "dnn": {"dense": True, "sparse": True},
            "linear": {"dense": True, "sparse": False},
        },
    )

    trainer = NeuralNetTrainer(
        # module and dataset configs
        module=WideAndDeep,
        module_params={
            "dnn_feature_defs": feature_defs["dnn"],
            "linear_feature_defs": feature_defs["linear"],
            "seed": rand_seed,
            "output_fn": torch.softmax,
            "output_fn_args": dict(dim=0),
            "dnn_dropout": 0.2,
        },
        dataset=datasets,
        dataset_options=torch_dataset_options,
        # trainer configs
        epochs=50,
        batch_size=32,
        loss_fn=LossFnStack(
            dict(fn=nn.BCELoss(), weight=0.2),
            dict(fn=nn.HingeEmbeddingLoss(), weight=0.8),
        ),
        optimizer=OptimizerStack(
            dict(cls=torch.optim.Adagrad, model_key="deep_model"),
            dict(
                cls=FTRL,
                args=dict(alpha=1.0, beta=1.0, l1=1.0, l2=1.0),
                model_key="wide_model",
            ),
        ),
        metric_fns=[
            torchmetrics.AUROC(),
            log_loss,
            torchmetrics.MeanSquaredError(squared=False),
        ],
        num_workers=num_workers,
        use_gpu=use_gpu,
        callbacks=["json", "tbx"],
    )
    results = trainer.run()
    pprint_results(results, print_interval=10)


if __name__ == "__main__":
    environ_validate(num_cpus=1 + 2)
    train_wdl_dist(num_workers=2)
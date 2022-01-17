import torch
import torch.nn as nn
from sklearn.metrics import log_loss

from phetware.model.torch import FNN
from phetware.util import pprint_results
from phetware.environ import environ_validate
from phetware import NeuralNetTrainer
from phetware.trainer import DefaultRun
from benchmarks.dataset import load_dataset_builtin


def train_fnn_dist(num_workers=2, use_gpu=False, rand_seed=2021):
    datasets, feature_defs, torch_dataset_options = load_dataset_builtin(
        dataset_name="criteo_mini",
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
            "output_fn": torch.sigmoid,
            "dnn_dropout": 0.2,
        },
        dataset=datasets,
        dataset_options=torch_dataset_options,
        # ddp configs
        ddp_options={"find_unused_parameters": True},
        # trainer configs
        epochs=10,
        batch_size=32,
        loss_fn=nn.BCELoss(),
        optimizer=torch.optim.Adam,
        metric_fns=[log_loss],
        num_workers=num_workers,
        use_gpu=use_gpu,
        callbacks=["json", "tbx"],
    )
    results = trainer.run(multi_runs=[DefaultRun, dict(module_params=dict(pre_trained_mode=False))])
    pprint_results(results)


if __name__ == "__main__":
    environ_validate(num_cpus=1 + 2)
    train_fnn_dist(num_workers=2)

import torch
import torch.nn as nn
import torchmetrics
from sklearn.metrics import log_loss

from phetware.model.torch import PNN
from phetware.util import pprint_results
from phetware.environ import environ_validate
from phetware import NeuralNetTrainer
from benchmarks.dataset import load_dataset_builtin


def train_pnn_dist(num_workers=2, use_gpu=False, rand_seed=2021):
    datasets, feature_defs, torch_dataset_options = load_dataset_builtin(
        dataset_name="criteo_mini",
        feature_def_settings={
            "dnn": {"dense": True, "sparse": True}})

    trainer = NeuralNetTrainer(
        # module and dataset configs
        module=PNN,
        module_params={
            "dnn_feature_defs": feature_defs["dnn"],
            "use_inner": True,
            "seed": rand_seed,
            "output_fn": torch.sigmoid,
            "dnn_dropout": 0.2},
        dataset=datasets,
        dataset_options=torch_dataset_options,
        # trainer configs
        epochs=10,
        batch_size=64,
        loss_fn=nn.BCELoss(),
        optimizer=torch.optim.Adam,
        metric_fns=[torchmetrics.AUROC(), log_loss],
        num_workers=num_workers,
        use_gpu=use_gpu,
        callbacks=["json", "tbx"]
    )
    results = trainer.run()
    pprint_results(results)


if __name__ == "__main__":
    environ_validate(num_cpus=1 + 2)
    train_pnn_dist(num_workers=2)

import torch
import torch.nn as nn
from torchmetrics.functional import auroc

from phetware.model.torch import DeepFM
from phetware.util import pprint_results
from phetware.environ import environ_validate
from phetware import NeuralNetTrainer
from examples.dataset import load_dataset


def train_deepfm_dist(num_workers=2, use_gpu=False, rand_seed=2021):
    datasets, feature_defs, torch_dataset_options = load_dataset(
        dataset_name="criteo_10k",
        feature_def_settings={
            "fm_1": {"dense": True, "sparse": True},
            "fm_2": {"dense": False, "sparse": True},
            "dnn": {"dense": True, "sparse": True},
        },
    )

    trainer = NeuralNetTrainer(
        # module and dataset configs
        module=DeepFM,
        module_params={
            "fm_1_feature_defs": feature_defs["fm_1"],
            "fm_2_feature_defs": feature_defs["fm_2"],
            "dnn_feature_defs": feature_defs["dnn"],
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
        optimizer=torch.optim.Adam,
        optimizer_args={
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
    optimizer=Adam
    early_stopping patience=2
    weight_decay=1e-3
    valid/BCELoss: 0.50389	valid/auroc: 0.73780
    """


if __name__ == "__main__":
    environ_validate(num_cpus=1 + 2)
    train_deepfm_dist(num_workers=2)

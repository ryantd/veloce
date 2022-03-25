import torch
import torch.nn as nn
from torchmetrics.functional import auroc

from veloce.model.ctr import WideAndDeep
from veloce.optimizer import OptimizerStack, FTRL
from veloce.util import pprint_results
from veloce import NeuralNetTrainer
from veloce.environ import environ_validate
from veloce.heterogeneous import PSStrategy, UpdateStrategy
from veloce.preprocessing import DataLoader


def train_wdl_dist(num_workers=2, use_gpu=False, rand_seed=2021):
    dataloader = DataLoader("examples/dataset/ctr/criteo_mini.txt")
    dataloader = (
        dataloader.set_label_column(label_name="label")
        .set_dense_features(feat_names=[f"I{i}" for i in range(1, 14)])
        .set_sparse_features(
            feat_names=[f"C{i}" for i in range(1, 27)], embedding_dim=1
        )
        # this order should follow the data file
        .set_features_order(order=("dense", "sparse"))
    )
    datasets = dataloader.split()
    dense_defs = dataloader.dense_defs
    sparse_defs = dataloader.sparse_defs
    torch_dataset_options = dataloader.gen_torch_dataset_options()

    trainer = NeuralNetTrainer(
        # module and dataset configs
        module=WideAndDeep,
        module_params={
            "dense_feature_defs": dense_defs,
            "sparse_feature_defs": sparse_defs,
            "dnn_dropout": 0.5,
            "dnn_hidden_units": (200, 200, 200),
            "seed": rand_seed,
        },
        dataset=datasets,
        dataset_options=torch_dataset_options,
        # trainer configs
        epochs=5,
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
        # set heterogeneous_strategy
        heterogeneous_strategy=PSStrategy(update_strategy=UpdateStrategy.Sync),
        num_workers=num_workers,
        use_gpu=use_gpu,
        callbacks=["json", "tbx"],
    )
    results = trainer.run()
    pprint_results(results)


if __name__ == "__main__":
    environ_validate(n_cpus=1 + 2)
    train_wdl_dist(num_workers=2)

import torch
import torch.nn as nn
from sklearn.metrics import log_loss

from phetware.util import pprint_results
from phetware.environ import environ_validate
from phetware.layer import OutputLayer
from phetware.model.torch.base import BaseModel, Linear
from phetware import NeuralNetTrainer
from benchmarks.dataset import load_dataset_builtin


class LR(BaseModel):
    def __init__(
        self,
        linear_feature_defs=None, l2_reg_linear=1e-5, seed=1024, device="cpu",
        output_fn=torch.sigmoid, output_fn_args=None, **kwargs
    ):
        super(LR, self).__init__(
            linear_feature_defs=linear_feature_defs,
            seed=seed,
            device=device,
        )
        self.linear = Linear(
            sparse_feature_defs=self.fds.linear_defs_sparse,
            dense_feature_defs=self.fds.linear_defs_dense,
            feature_named_index_mapping=self.feature_name_to_index,
            device=device,
        )
        self.add_regularization_weight(self.linear.parameters(), l2=l2_reg_linear)
        self.output = OutputLayer(output_fn=output_fn, output_fn_args=output_fn_args)
        self.to(device)

    def forward(self, X):
        logit = self.linear(X)
        return self.output(logit)


def train_lr_dist(num_workers=2, use_gpu=False, rand_seed=2021):
    datasets, feature_defs, torch_dataset_options = load_dataset_builtin(
        dataset_name="criteo_mini",
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
        epochs=10,
        batch_size=32,
        loss_fn=nn.BCELoss(),
        optimizer=torch.optim.Adam,
        metric_fns=[log_loss],
        num_workers=num_workers,
        use_gpu=use_gpu,
        callbacks=["json", "tbx"],
    )
    results = trainer.run()
    pprint_results(results)


if __name__ == "__main__":
    environ_validate(num_cpus=1 + 2)
    train_lr_dist(num_workers=2)

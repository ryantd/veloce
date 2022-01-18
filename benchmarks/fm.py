import torch
import torch.nn as nn
from sklearn.metrics import log_loss

from phetware.util import pprint_results
from phetware.environ import environ_validate
from phetware.layer import OutputLayer, FM as FMLayer
from phetware.model.torch.base import BaseModel
from phetware.inputs import (embedding_dict_gen, collect_inputs_and_embeddings, concat_dnn_inputs)
from phetware import NeuralNetTrainer
from benchmarks.dataset import load_dataset_builtin


class FM(BaseModel):
    def __init__(
        self,
        fm_feature_defs=None, l2_reg_fm=1e-5, seed=1024, device="cpu",
        output_fn=torch.sigmoid, output_fn_args=None, init_std=0.0001, **kwargs
    ):
        super(FM, self).__init__(
            fm_feature_defs=fm_feature_defs,
            seed=seed,
            device=device,
        )
        self.fm = FMLayer()
        self.fm_embedding_layer = embedding_dict_gen(
            self.fds.fm_defs_sparse,
            init_std=init_std,
            sparse=False,
            device=device,
        )
        self.add_regularization_weight(
            self.fm_embedding_layer.parameters(), l2=l2_reg_fm
        )
        self.output = OutputLayer(output_fn=output_fn, output_fn_args=output_fn_args)
        self.to(device)

    def forward(self, X):
        dense_vals, sparse_embs = collect_inputs_and_embeddings(
            X,
            sparse_feature_defs=self.fds.fm_defs_sparse,
            dense_feature_defs=self.fds.fm_defs_dense,
            feature_name_to_index=self.feature_name_to_index,
            embedding_layer_def=self.fm_embedding_layer,
        )
        fm_input = concat_dnn_inputs(sparse_embs, dense_vals).unsqueeze(dim=1)
        return self.output(self.fm(fm_input))


def train_fm_dist(num_workers=2, use_gpu=False, rand_seed=2021):
    datasets, feature_defs, torch_dataset_options = load_dataset_builtin(
        dataset_name="criteo_mini",
        feature_def_settings={
            "fm": {"dense": True, "sparse": True},
        },
    )

    trainer = NeuralNetTrainer(
        # module and dataset configs
        module=FM,
        module_params={
            "fm_feature_defs": feature_defs["fm"],
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
    train_fm_dist(num_workers=2)

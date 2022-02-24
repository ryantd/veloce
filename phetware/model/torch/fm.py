import torch

from phetware.layer import OutputLayer, FMNative
from phetware.model.torch.base import BaseModel
from phetware.inputs import (
    embedding_dict_gen,
    collect_inputs_and_embeddings,
    concat_inputs,
)


class FM(BaseModel):
    def __init__(
        self,
        fm_feature_defs=None,
        k_factor=10,
        l2_reg_fm=1e-3,
        l2_reg_embedding=1e-3,
        fm_dropout=0,
        seed=1024,
        device="cpu",
        output_fn=torch.sigmoid,
        output_fn_args=None,
        init_std=1e-4,
    ):
        super(FM, self).__init__(
            fm_feature_defs=fm_feature_defs,
            seed=seed,
            device=device,
        )
        self.embedding_layer = embedding_dict_gen(
            self.fds.fm_defs_sparse,
            init_std=init_std,
            sparse=False,
            device=device,
        )
        self.fm = FMNative(
            feature_def_dims=sum(fc.dimension for fc in self.fds.fm_defs_dense)
            + sum(fc.embedding_dim for fc in self.fds.fm_defs_sparse),
            k_factor=k_factor,
            dropout_rate=fm_dropout,
            init_std=init_std,
        )
        self.add_regularization_weight(
            self.embedding_layer.parameters(), l2=l2_reg_embedding
        )
        self.add_regularization_weight(self.fm.parameters(), l2=l2_reg_fm)
        self.output = OutputLayer(output_fn=output_fn, output_fn_args=output_fn_args)
        self.to(device)

    def forward(self, X):
        dense_vals, sparse_embs = collect_inputs_and_embeddings(
            X,
            sparse_feature_defs=self.fds.fm_defs_sparse,
            dense_feature_defs=self.fds.fm_defs_dense,
            feature_name_to_index=self.feature_name_to_index,
            embedding_layer_def=self.embedding_layer,
        )
        logit = self.fm(concat_inputs(sparse_embs, dense_vals))
        return self.output(logit)

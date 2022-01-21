import torch.nn as nn
import torch

from phetware.layer import DNN, OutputLayer, FMNative
from phetware.inputs import (
    concat_inputs,
    compute_inputs_dim,
    embedding_dict_gen,
    collect_inputs_and_embeddings,
)
from .base import BaseModel


class FNN(BaseModel):
    def __init__(
        self,
        # feature_defs
        dnn_feature_defs=None,
        # fm related
        pre_trained_mode=False,
        k_factor=10,
        fm_dropout=0,
        l2_reg_fm=1e-3,
        # dnn related
        dnn_hidden_units=(256, 128),
        dnn_use_bn=False,
        dnn_activation="relu",
        l2_reg_embedding=1e-3,
        l2_reg_dnn=1e-3,
        dnn_dropout=0,
        # base config
        seed=1024,
        output_fn=torch.sigmoid,
        output_fn_args=None,
        device="cpu",
        init_std=1e-4,
        **kwargs
    ):
        super(FNN, self).__init__(
            dnn_feature_defs=dnn_feature_defs,
            seed=seed,
            device=device,
        )
        self.use_dnn = len(dnn_feature_defs) > 0 and len(dnn_hidden_units) > 0
        self.pre_trained_mode = pre_trained_mode

        # fm layers setup
        self.fm = FMNative(
            feature_def_dims=sum(fc.dimension for fc in self.fds.dnn_defs_dense)
            + sum(fc.embedding_dim for fc in self.fds.dnn_defs_sparse),
            k_factor=k_factor,
            dropout_rate=fm_dropout,
            init_std=init_std,
        )
        self.add_regularization_weight(self.fm.parameters(), l2=l2_reg_fm)

        # dnn layer setup
        if self.use_dnn:
            # embedding layer
            self.dnn_embedding_layer = embedding_dict_gen(
                self.fds.dnn_defs_sparse, init_std=init_std, sparse=False, device=device
            )
            self.add_regularization_weight(
                self.dnn_embedding_layer.parameters(), l2=l2_reg_embedding
            )
            self.dnn = DNN(
                compute_inputs_dim(
                    sparse_feature_defs=self.fds.dnn_defs_sparse,
                    dense_feature_defs=self.fds.dnn_defs_dense,
                ),
                dnn_hidden_units,
                activation=dnn_activation,
                l2_reg=l2_reg_dnn,
                dropout_rate=dnn_dropout,
                use_bn=dnn_use_bn,
                init_std=init_std,
                device=device,
            )
            self.final_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False).to(
                device
            )
            self.add_regularization_weight(
                filter(
                    lambda x: "weight" in x[0] and "bn" not in x[0],
                    self.dnn.named_parameters(),
                ),
                l2=l2_reg_dnn,
            )
            self.add_regularization_weight(self.final_linear.weight, l2=l2_reg_dnn)

        # output layer setup
        self.output = OutputLayer(output_fn=output_fn, output_fn_args=output_fn_args)
        self.to(device)

    def forward(self, X):
        dnn_dense_vals, dnn_sparse_embs = collect_inputs_and_embeddings(
            X,
            sparse_feature_defs=self.fds.dnn_defs_sparse,
            dense_feature_defs=self.fds.dnn_defs_dense,
            feature_name_to_index=self.feature_name_to_index,
            embedding_layer_def=self.dnn_embedding_layer,
        )
        if self.pre_trained_mode:
            logit = self.fm(concat_inputs(dnn_sparse_embs, dnn_dense_vals))
        elif self.use_dnn:
            output = self.dnn(concat_inputs(dnn_sparse_embs, dnn_dense_vals))
            logit = self.final_linear(output)
        y = self.output(logit)
        return y

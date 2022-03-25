import torch.nn as nn
import torch

from veloce.layer import DNN, OutputLayer, FMNative
from veloce.inputs import (
    concat_inputs,
    compute_inputs_dim,
    embedding_dict_gen,
    collect_inputs_and_embeddings,
)
from .base import BaseModel, Linear


class WideAndDeep(BaseModel):
    def __init__(
        self,
        # feature defs
        dense_feature_defs=None,
        sparse_feature_defs=None,
        # wide related
        use_fm=False,
        k_factor=10,
        fm_dropout=0,
        l2_reg_linear=1e-3,
        # dnn related
        dnn_hidden_units=(256, 128),
        dnn_use_bn=False,
        dnn_activation="relu",
        dnn_dropout=0,
        l2_reg_embedding=1e-3,
        l2_reg_dnn=1e-3,
        # base configs
        seed=1024,
        output_fn=torch.sigmoid,
        output_fn_args=None,
        device="cpu",
        init_std=1e-4,
    ):
        super(WideAndDeep, self).__init__(seed=seed, device=device)
        self.dense_defs = dense_feature_defs
        self.sparse_defs = sparse_feature_defs
        self.use_dnn = len(dnn_hidden_units) > 0
        self.use_fm = use_fm

        # embedding layer setup
        self.dnn_embedding_layer = embedding_dict_gen(
            self.sparse_defs, init_std=init_std, sparse=False, device=device
        )
        self.add_regularization_weight(
            self.dnn_embedding_layer.parameters(), l2=l2_reg_embedding
        )

        # wide model setup
        if self.use_fm:
            self.wide_model = FMNative(
                feature_def_dims=sum(fc.dimension for fc in self.dense_defs)
                + sum(fc.embedding_dim for fc in self.sparse_defs),
                k_factor=k_factor,
                dropout_rate=fm_dropout,
                init_std=init_std,
            )
        else:
            self.wide_model = Linear(
                sparse_feature_defs=self.sparse_defs,
                dense_feature_defs=self.dense_defs,
                device=device,
            )
        self.add_regularization_weight(self.wide_model.parameters(), l2=l2_reg_linear)

        # deep model setup
        if self.use_dnn:
            self.deep_model = DNN(
                compute_inputs_dim(
                    sparse_feature_defs=self.sparse_defs,
                    dense_feature_defs=self.dense_defs,
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
                    self.deep_model.named_parameters(),
                ),
                l2=l2_reg_dnn,
            )
            self.add_regularization_weight(self.final_linear.weight, l2=l2_reg_dnn)

        # output layer setup
        self.output = OutputLayer(output_fn=output_fn, output_fn_args=output_fn_args)

    def forward(self, X):
        if self.use_fm:
            dense_values, sparse_embeddings = collect_inputs_and_embeddings(
                X,
                sparse_feature_defs=self.sparse_defs,
                dense_feature_defs=self.dense_defs,
                embedding_layer_def=self.dnn_embedding_layer,
            )
            wide_input = concat_inputs(sparse_embeddings, dense_values)
            logit = self.wide_model(wide_input)
        else:
            logit = self.wide_model(X)
        if self.use_dnn:
            dense_values, sparse_embeddings = collect_inputs_and_embeddings(
                X,
                sparse_feature_defs=self.sparse_defs,
                dense_feature_defs=self.dense_defs,
                embedding_layer_def=self.dnn_embedding_layer,
            )
            dnn_input = concat_inputs(sparse_embeddings, dense_values)
            dnn_output = self.deep_model(dnn_input)
            dnn_logit = self.final_linear(dnn_output)
            logit += dnn_logit
        y = self.output(logit)
        return y

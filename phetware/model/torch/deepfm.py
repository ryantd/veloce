import torch.nn as nn
import torch

from phetware.layer import DNN, OutputLayer, FM
from phetware.inputs import concat_dnn_inputs, compute_inputs_dim, embedding_dict_gen, collect_inputs_and_embeddings
from .base import BaseModel, Linear


class DeepFM(BaseModel):
    def __init__(
        self,
        # feature_defs
        fm_1_feature_defs, fm_2_feature_defs, dnn_feature_defs,
        # fm related
        use_fm=True, l2_reg_fm_1=1e-5, l2_reg_fm_2=1e-5,
        # dnn related
        dnn_hidden_units=(256, 128), dnn_use_bn=False, dnn_activation="relu",
        l2_reg_embedding=1e-5, l2_reg_dnn=0, dnn_dropout=0,
        # base config
        seed=1024, output_fn=torch.sigmoid, output_fn_args=None, device="cpu",
        init_std=0.0001,
    ):
        super(DeepFM, self).__init__(
            fm_1_feature_defs=fm_1_feature_defs,
            fm_2_feature_defs=fm_2_feature_defs,
            dnn_feature_defs=dnn_feature_defs, seed=seed, device=device)
        self.use_dnn = len(dnn_feature_defs) > 0 and len(dnn_hidden_units) > 0
        self.use_fm = use_fm

        # fm layers setup
        if self.use_fm:
            self.fm_1 = Linear(
                sparse_feature_defs=self.fds.fm_1_defs_sparse,
                dense_feature_defs=self.fds.fm_1_defs_dense,
                feature_named_index_mapping=self.feature_name_to_index,
                device=device)
            self.add_regularization_weight(
                self.fm_1.parameters(), l2=l2_reg_fm_1)

            self.fm_2 = FM()
            # fm_2 embedding layer
            self.fm_2_embedding_layer = embedding_dict_gen(
                self.fds.fm_2_defs_sparse,
                init_std=init_std, sparse=False, device=device)
            self.add_regularization_weight(
                self.fm_2_embedding_layer.parameters(), l2=l2_reg_fm_2)

        # dnn layer setup
        if self.use_dnn:
            # embedding layer
            self.dnn_embedding_layer = embedding_dict_gen(
                self.fds.dnn_defs_sparse,
                init_std=init_std, sparse=False, device=device)
            self.add_regularization_weight(
                self.dnn_embedding_layer.parameters(), l2=l2_reg_embedding)
        
            self.dnn = DNN(
                compute_inputs_dim(
                    sparse_feature_defs=self.fds.dnn_defs_sparse,
                    dense_feature_defs=self.fds.dnn_defs_dense),
                dnn_hidden_units,
                activation=dnn_activation,
                l2_reg=l2_reg_dnn,
                dropout_rate=dnn_dropout,
                use_bn=dnn_use_bn,
                init_std=init_std,
                device=device)
            self.final_linear = nn.Linear(
                dnn_hidden_units[-1], 1, bias=False).to(device)

            self.add_regularization_weight(filter(
                lambda x: 'weight' in x[0] and 'bn' not in x[0],
                self.dnn.named_parameters()), l2=l2_reg_dnn)
            self.add_regularization_weight(
                self.final_linear.weight, l2=l2_reg_dnn)

        # output layer setup
        self.output = OutputLayer(
            output_fn=output_fn, output_fn_args=output_fn_args)
        self.to(device)

    def forward(self, X):
        if self.use_fm:
            # fm_1
            logit = self.fm_1(X)
            # fm_2
            fm_2_sparse_embs = collect_inputs_and_embeddings(
                X, sparse_feature_defs=self.fds.fm_2_defs_sparse,
                feature_name_to_index=self.feature_name_to_index,
                embedding_layer_def=self.fm_2_embedding_layer,
                return_dense=False)
            fm_2_input = torch.cat(fm_2_sparse_embs, dim=1)
            logit += self.fm_2(fm_2_input)

        # dnn
        if self.use_dnn:
            dnn_dense_vals, dnn_sparse_embs = collect_inputs_and_embeddings(
                X, sparse_feature_defs=self.fds.dnn_defs_sparse,
                dense_feature_defs=self.fds.dnn_defs_dense,
                feature_name_to_index=self.feature_name_to_index,
                embedding_layer_def=self.dnn_embedding_layer)
            dnn_input = concat_dnn_inputs(dnn_sparse_embs, dnn_dense_vals)
            dnn_output = self.dnn(dnn_input)
            dnn_logit = self.final_linear(dnn_output)
            logit += dnn_logit
        y = self.output(logit)
        return y

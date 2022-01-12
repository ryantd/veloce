import torch.nn as nn
import torch

from phetware.layer import DNN, OutputLayer, InnerProduct, OutterProduct
from phetware.inputs import concat_dnn_inputs, compute_inputs_dim, embedding_dict_gen, collect_inputs_and_embeddings, concat_inputs
from .base import BaseModel


class PNN(BaseModel):
    def __init__(
        self,
        # feature_defs
        dnn_feature_defs,
        # specific config
        use_inner=True, use_outter=False, outter_kernel_type='mat',
        # dnn related
        dnn_hidden_units=(256, 128), dnn_use_bn=False, dnn_activation="relu",
        l2_reg_embedding=1e-5, l2_reg_dnn=0, dnn_dropout=0,
        # base config
        seed=1024, output_fn=torch.sigmoid, output_fn_args=None, device="cpu",
        init_std=0.0001,
    ):
        super(PNN, self).__init__(
            dnn_feature_defs=dnn_feature_defs, seed=seed, device=device)
        if outter_kernel_type not in ['mat', 'vec', 'num']:
            raise ValueError("Arg kernel_type must be mat, vec or num")

        self.use_dnn = len(dnn_feature_defs) > 0 and len(dnn_hidden_units) > 0
        self.use_inner = use_inner
        self.use_outter = use_outter
        self.outter_kernel_type = outter_kernel_type

        product_out_dim = 0
        inputs_dim = compute_inputs_dim(
            self.fds.dnn_defs_sparse, feature_group=True)
        num_pairs = int(inputs_dim * (inputs_dim - 1) / 2)

        # product layer setup
        if self.use_inner:
            product_out_dim += num_pairs
            self.inner_product = InnerProduct(device=device)

        if self.use_outter:
            product_out_dim += num_pairs
            self.outter_product = OutterProduct(
                inputs_dim, self.embedding_size,
                kernel_type=self.outter_kernel_type, device=device)

        # dnn layer setup
        if self.use_dnn:
            # embedding layer
            self.dnn_embedding_layer = embedding_dict_gen(
                self.fds.dnn_defs_sparse,
                init_std=init_std, sparse=False, device=device)
            self.add_regularization_weight(
                self.dnn_embedding_layer.parameters(), l2=l2_reg_embedding)
        
            self.dnn = DNN(
                product_out_dim + compute_inputs_dim(
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
        dnn_dense_vals, dnn_sparse_embs = collect_inputs_and_embeddings(
            X, sparse_feature_defs=self.fds.dnn_defs_sparse,
            dense_feature_defs=self.fds.dnn_defs_dense,
            feature_name_to_index=self.feature_name_to_index,
            embedding_layer_def=self.dnn_embedding_layer)
        
        linear_signal = torch.flatten(
            concat_inputs(dnn_sparse_embs), start_dim=1)
        if self.use_inner:
            inner_product = torch.flatten(
                self.inner_product(dnn_sparse_embs), start_dim=1)
        if self.use_outter:
            outer_product = self.outter_product(dnn_sparse_embs)

        if self.use_outter and self.use_inner:
            product_layer = torch.cat(
                [linear_signal, inner_product, outer_product], dim=1)
        elif self.use_outter:
            product_layer = torch.cat([linear_signal, outer_product], dim=1)
        elif self.use_inner:
            product_layer = torch.cat([linear_signal, inner_product], dim=1)
        else:
            product_layer = linear_signal

        # dnn
        if self.use_dnn:
            dnn_input = concat_dnn_inputs([product_layer], dnn_dense_vals)
            dnn_output = self.dnn(dnn_input)
            dnn_logit = self.final_linear(dnn_output)
            logit = dnn_logit
        y = self.output(logit)
        return y

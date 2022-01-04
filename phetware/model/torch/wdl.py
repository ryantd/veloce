import torch.nn as nn

from phetware.layer import DNN, OutputLayer
from phetware.inputs import concat_dnn_inputs, compute_inputs_dim, embedding_dict_gen, collect_inputs_and_embeddings
from .base import BaseModel, Linear


class WideAndDeep(BaseModel):
    def __init__(
        self,
        linear_feature_columns, dnn_feature_columns, dnn_hidden_units=(256, 128),
        l2_reg_linear=1e-5, l2_reg_embedding=1e-5, l2_reg_dnn=0,
        init_std=0.0001, seed=1024, dnn_dropout=0, dnn_activation='relu',
        dnn_use_bn=False, task='binary', device='cpu'
    ):
        super(WideAndDeep, self).__init__(
            linear_feature_columns, dnn_feature_columns, seed=seed,
            device=device)
        self.use_dnn = len(dnn_feature_columns) > 0 and len(dnn_hidden_units) > 0

        # embedding layer setup
        self.dnn_embedding_layer = embedding_dict_gen(
            self.fcs.dnn_sparse_fcs,
            init_std=init_std, sparse=False,
            device=device)
        self.add_regularization_weight(
            self.dnn_embedding_layer.parameters(), l2=l2_reg_embedding)

        # wide model setup
        self.wide_model = Linear(
            sparse_feature_columns=self.fcs.linear_sparse_fcs,
            dense_feature_columns=self.fcs.linear_dence_fcs,
            feature_named_index_mapping=self.feature_name_to_index,
            device=device)
        self.add_regularization_weight(
            self.wide_model.parameters(), l2=l2_reg_linear)
        
        # deep model setup
        if self.use_dnn:
            self.deep_model = DNN(
                compute_inputs_dim(
                    sparse_feature_columns=self.fcs.dnn_sparse_fcs,
                    dense_feature_columns=self.fcs.dnn_dence_fcs),
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
                self.deep_model.named_parameters()), l2=l2_reg_dnn)
            self.add_regularization_weight(
                self.final_linear.weight, l2=l2_reg_dnn)

        # output layer setup
        self.output = OutputLayer(task)
        self.to(device)

    def forward(self, X):
        dense_values, sparse_embeddings = collect_inputs_and_embeddings(
            X, sparse_feature_columns=self.fcs.dnn_sparse_fcs,
            dense_feature_columns=self.fcs.dnn_dence_fcs,
            feature_name_to_index=self.feature_name_to_index,
            embedding_layer_def=self.dnn_embedding_layer)
        logit = self.wide_model(X)

        if self.use_dnn:
            dnn_input = concat_dnn_inputs(sparse_embeddings, dense_values)
            dnn_output = self.deep_model(dnn_input)
            dnn_logit = self.final_linear(dnn_output)
            logit += dnn_logit
        y = self.output(logit)
        return y

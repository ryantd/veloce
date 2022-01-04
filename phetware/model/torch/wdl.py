import torch.nn as nn

from phetware.layer import DNN
from phetware.inputs import concat_dnn_inputs, compute_input_dim
from .base import BaseModel


class WideAndDeep(BaseModel):
    def __init__(
        self,
        linear_feature_columns, dnn_feature_columns, dnn_hidden_units=(256, 128),
        l2_reg_linear=1e-5, l2_reg_embedding=1e-5, l2_reg_dnn=0,
        init_std=0.0001, seed=1024, dnn_dropout=0, dnn_activation='relu',
        dnn_use_bn=False, task='binary', device='cpu'
    ):
        super(WideAndDeep, self).__init__(
            linear_feature_columns, dnn_feature_columns,
            l2_reg_linear=l2_reg_linear, l2_reg_embedding=l2_reg_embedding,
            init_std=init_std, task=task, device=device)
        
        self.use_dnn = len(dnn_feature_columns) > 0 and len(dnn_hidden_units) > 0
        
        if self.use_dnn:
            self.dnn_model = DNN(
                compute_input_dim(
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
                self.dnn_model.named_parameters()), l2=l2_reg_dnn)
            self.add_regularization_weight(
                self.final_linear.weight, l2=l2_reg_dnn)
        self.to(device)

    def forward(self, X):
        sparse_embeddings, dense_values = self.fetch_dnn_inputs(X)
        logit = self.linear_model(X)

        if self.use_dnn:
            dnn_input = concat_dnn_inputs(sparse_embeddings, dense_values)
            dnn_output = self.dnn_model(dnn_input)
            dnn_logit = self.final_linear(dnn_output)
            logit += dnn_logit
        y = self.output(logit)
        return y

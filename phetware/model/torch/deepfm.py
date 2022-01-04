import torch
import torch.nn as nn
from torch.nn.modules import sparse

from phetware.layer import FM, DNN
from phetware.inputs import concat_dnn_inputs, compute_inputs_dim
from .base import BaseModel


class DeepFM(BaseModel):
    def __init__(
        self,
        linear_feature_columns, dnn_feature_columns, use_fm=True,
        dnn_hidden_units=(256, 128), l2_reg_linear=0.00001,
        l2_reg_embedding=0.00001, l2_reg_dnn=0, init_std=0.0001, seed=1024,
        dnn_dropout=0, dnn_activation='relu', dnn_use_bn=False, task='binary',
        device='cpu'
    ):
        super(DeepFM, self).__init__(
            linear_feature_columns, dnn_feature_columns, l2_reg_linear,
            l2_reg_embedding, init_std, seed, task, device)
        
        self.use_fm = use_fm
        self.use_dnn = len(dnn_feature_columns) > 0 and len(dnn_hidden_units) > 0
        
        if use_fm:
            self.fm = FM()
        
        if self.use_dnn:
            self.dnn_model = DNN(
                compute_inputs_dim(
                    sparse_feature_columns=self.fcs.dnn_sparse_fcs,
                    dense_feature_columns=self.fcs.dnn_dence_fcs),
                dnn_hidden_units,
                activation=dnn_activation, l2_reg=l2_reg_dnn,
                dropout_rate=dnn_dropout, use_bn=dnn_use_bn, init_std=init_std,
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

        if self.use_fm and len(sparse_embeddings) > 0:
            fm_input = torch.cat(sparse_embeddings, dim=1)
            logit += self.fm(fm_input)

        if self.use_dnn:
            dnn_input = concat_dnn_inputs(
                sparse_embeddings, dense_values)
            dnn_output = self.dnn_model(dnn_input)
            dnn_logit = self.final_linear(dnn_output)
            logit += dnn_logit
        y = self.output(logit)
        return y

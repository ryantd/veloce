import torch
import torch.nn as nn

from phetware.inputs import build_embedding_dict, build_feature_named_index_mapping
from phetware.layer import OutputLayer
from phetware.feature_column import FeatureColumnSet


class BaseModel(nn.Module):
    def __init__(
        self, linear_feature_columns, dnn_feature_columns, l2_reg_linear=1e-5,
        l2_reg_embedding=1e-5, init_std=0.0001, seed=1024, task='binary',
        device='cpu'
    ):
        super(BaseModel, self).__init__()
        self.device = device
        
        self.fcs = FeatureColumnSet(
            linear_feature_columns=linear_feature_columns,
            dnn_feature_columns=dnn_feature_columns)
        self.fcs.sorter()
        self.feature_name_to_index = build_feature_named_index_mapping(
            self.fcs.all_fcs)
        self.dnn_embedding_dict = build_embedding_dict(
            self.fcs.dnn_sparse_fcs,
            init_std=init_std, sparse=False,
            device=device)
        self.linear_model = Linear(
            sparse_feature_columns=self.fcs.linear_sparse_fcs,
            dense_feature_columns=self.fcs.linear_dence_fcs,
            feature_named_index_mapping=self.feature_name_to_index,
            device=device)
        
        self.regularization_weight = []
        self.add_regularization_weight(
            self.dnn_embedding_dict.parameters(), l2=l2_reg_embedding)
        self.add_regularization_weight(
            self.linear_model.parameters(), l2=l2_reg_linear)

        self.output = OutputLayer(task)
        self.to(device)

    def add_regularization_weight(self, weight_list, l1=0.0, l2=0.0):
        if isinstance(weight_list, torch.nn.parameter.Parameter):
            weight_list = [weight_list]
        else:
            weight_list = list(weight_list)
        self.regularization_weight.append((weight_list, l1, l2))
        return self.regularization_weight
    
    def fetch_dnn_inputs(self, X, support_dense=True):
        sparse_feature_columns = self.fcs.dnn_sparse_fcs
        dense_feature_columns = self.fcs.dnn_dence_fcs

        if not support_dense and len(dense_feature_columns) > 0:
            raise ValueError(
                "DenseFeat is not supported in dnn_feature_columns")

        sparse_embeddings = [self.dnn_embedding_dict[feat.embedding_name](X[
            :, self.feature_name_to_index[feat.name][0]: self.feature_name_to_index[feat.name][1]
        ].long()) for feat in sparse_feature_columns]
        dense_values = [X[
            :, self.feature_name_to_index[feat.name][0]: self.feature_name_to_index[feat.name][1]
        ] for feat in dense_feature_columns]
        return sparse_embeddings, dense_values


class Linear(nn.Module):
    def __init__(
        self, sparse_feature_columns, dense_feature_columns,
        feature_named_index_mapping, init_std=0.0001, device='cpu'
    ):
        super(Linear, self).__init__()
        self.feature_name_to_index = feature_named_index_mapping
        self.device = device
        self.sparse_feature_columns = sparse_feature_columns
        self.dense_feature_columns = dense_feature_columns

        self.linear_embedding_dict = build_embedding_dict(
            self.sparse_feature_columns,
            init_std, linear=True, sparse=False,
            device=device)

        if len(self.dense_feature_columns):
            self.weight = nn.Parameter(
                torch.Tensor(
                    sum(fc.dimension for fc in self.dense_feature_columns),
                    1).to(device))
            torch.nn.init.normal_(self.weight, mean=0, std=init_std)

    def forward(self, X, sparse_feat_refine_weight=None):
        sparse_embeddings = [self.linear_embedding_dict[feat.embedding_name](X[
            :, self.feature_name_to_index[feat.name][0]:self.feature_name_to_index[feat.name][1]
        ].long()) for feat in self.sparse_feature_columns]
        dense_values = [X[
            :, self.feature_name_to_index[feat.name][0]:self.feature_name_to_index[feat.name][1]
        ] for feat in self.dense_feature_columns]

        linear_logit = torch.zeros(
            [X.shape[0], 1]).to(sparse_embeddings[0].device)
        if len(sparse_embeddings) > 0:
            sparse_embedding_cat = torch.cat(sparse_embeddings, dim=-1)
            if sparse_feat_refine_weight is not None:
                sparse_embedding_cat = sparse_embedding_cat * sparse_feat_refine_weight.unsqueeze(1)
            sparse_feat_logit = torch.sum(sparse_embedding_cat, dim=-1, keepdim=False)
            linear_logit += sparse_feat_logit
        if len(dense_values) > 0:
            dense_value_logit = torch.cat(dense_values, dim=-1).matmul(self.weight)
            linear_logit += dense_value_logit
        return linear_logit

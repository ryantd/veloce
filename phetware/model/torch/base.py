import torch
import torch.nn as nn

from phetware.inputs import embedding_dict_gen, build_feature_named_index_mapping, collect_inputs_and_embeddings
from phetware.feature_column import FeatureDefSet


class BaseModel(nn.Module):
    def __init__(self, **kwargs):
        super(BaseModel, self).__init__()
        torch.manual_seed(kwargs["seed"])
        self.device = kwargs["device"]
        self.regularization_weight = []
        del kwargs["seed"]
        del kwargs["device"]

        # produce dense and sparse fds on dnn and linear inputs respectively
        self.fds = FeatureDefSet(kwargs)
        self.fds.sorter()

        # produce mapping from feature names to columns index
        self.feature_name_to_index = build_feature_named_index_mapping(
            self.fds.all_defs)

        self.to(self.device)

    def add_regularization_weight(self, weight_list, l1=0.0, l2=0.0):
        if isinstance(weight_list, torch.nn.parameter.Parameter):
            weight_list = [weight_list]
        else:
            weight_list = list(weight_list)
        self.regularization_weight.append((weight_list, l1, l2))
        return self.regularization_weight


class Linear(nn.Module):
    def __init__(
        self, sparse_feature_defs, dense_feature_defs,
        feature_named_index_mapping, init_std=0.0001, device='cpu'
    ):
        super(Linear, self).__init__()
        self.feature_name_to_index = feature_named_index_mapping
        self.device = device
        self.sparse_feature_defs = sparse_feature_defs
        self.dense_feature_defs = dense_feature_defs

        self.linear_embedding_dict = embedding_dict_gen(
            self.sparse_feature_defs,
            init_std, linear=True, sparse=False,
            device=device)

        if len(self.dense_feature_defs):
            self.weight = nn.Parameter(
                torch.Tensor(
                    sum(fc.dimension for fc in self.dense_feature_defs),
                    1).to(device))
            torch.nn.init.normal_(self.weight, mean=0, std=init_std)

    def forward(self, X, sparse_feat_refine_weight=None):
        dense_values, sparse_embeddings = collect_inputs_and_embeddings(
            X, sparse_feature_defs=self.sparse_feature_defs,
            dense_feature_defs=self.dense_feature_defs,
            feature_name_to_index=self.feature_name_to_index,
            embedding_layer_def=self.linear_embedding_dict)

        linear_logit = torch.zeros([X.shape[0], 1]).to(self.device)
        if len(sparse_embeddings) > 0:
            linear_logit = linear_logit.to(sparse_embeddings[0].device)
            sparse_embedding_cat = torch.cat(sparse_embeddings, dim=-1)
            if sparse_feat_refine_weight is not None:
                sparse_embedding_cat = sparse_embedding_cat * sparse_feat_refine_weight.unsqueeze(1)
            sparse_feat_logit = torch.sum(sparse_embedding_cat, dim=-1, keepdim=False)
            linear_logit += sparse_feat_logit
        if len(dense_values) > 0:
            dense_value_logit = torch.cat(dense_values, dim=-1).matmul(self.weight)
            linear_logit += dense_value_logit
        return linear_logit

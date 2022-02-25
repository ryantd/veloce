import torch
import torch.nn as nn

DEFAULT_GROUP_NAME = "default_group"


class SparseFeat(object):
    key = "sparse"

    def __init__(
        self,
        name,
        vocabulary_size,
        column_idx,
        *,
        embedding_dim=4,
        dtype="int32",
        group_name=DEFAULT_GROUP_NAME,
        feat_type="SparseFeat",
    ):
        if embedding_dim == "auto":
            embedding_dim = 6 * int(pow(vocabulary_size, 0.25))
        self.name = name
        self.vocabulary_size = int(vocabulary_size)
        self.column_idx = int(column_idx)
        self.embedding_dim = int(embedding_dim)
        self.dtype = dtype
        self.group_name = group_name
        self.feat_type = feat_type


class DenseFeat(object):
    key = "dense"

    def __init__(
        self, name, column_idx, *, dimension=1, dtype="float32", feat_type="DenseFeat"
    ):
        self.name = name
        self.column_idx = int(column_idx)
        self.dimension = int(dimension)
        self.dtype = dtype
        self.feat_type = feat_type


def find_feature_values(value):
    try:
        return all(["feat_type" in v for v in value])
    except:
        return False


def rebuild_feature_values(feature_values):
    new_feature_values = list()
    for feat in feature_values:
        feat_variant = globals()[feat["feat_type"]]
        new_feature_values.append(feat_variant(**feat))
    return new_feature_values


def concat_inputs(sparse_embedding_list, dense_value_list):
    if len(sparse_embedding_list) > 0 and len(dense_value_list) > 0:
        sparse_dnn_input = torch.flatten(
            torch.cat(sparse_embedding_list, dim=-1), start_dim=1
        )
        dense_dnn_input = torch.flatten(
            torch.cat(dense_value_list, dim=-1), start_dim=1
        )
        return concatenate([sparse_dnn_input, dense_dnn_input])
    elif len(sparse_embedding_list) > 0:
        return torch.flatten(torch.cat(sparse_embedding_list, dim=-1), start_dim=1)
    elif len(dense_value_list) > 0:
        return torch.flatten(torch.cat(dense_value_list, dim=-1), start_dim=1)
    else:
        raise NotImplementedError


def embedding_dict_gen(
    sparse_feature_defs, init_std=0.0001, linear=False, sparse=False, device="cpu"
):
    embedding_dict = nn.ModuleDict(
        {
            feat.name: nn.Embedding(
                feat.vocabulary_size,
                feat.embedding_dim if not linear else 1,
                sparse=sparse,
            )
            for feat in sparse_feature_defs
        }
    )

    for tensor in embedding_dict.values():
        nn.init.normal_(tensor.weight, mean=0, std=init_std)

    return embedding_dict.to(device)


def compute_inputs_dim(
    sparse_feature_defs=None, dense_feature_defs=None, feature_group=False
):
    input_dim = 0
    if sparse_feature_defs is not None:
        if feature_group:
            sparse_input_dim = len(sparse_feature_defs)
        else:
            sparse_input_dim = sum(feat.embedding_dim for feat in sparse_feature_defs)
        input_dim += sparse_input_dim
    if dense_feature_defs is not None:
        dense_input_dim = sum(map(lambda x: x.dimension, dense_feature_defs))
        input_dim += dense_input_dim
    return input_dim


def collect_inputs_and_embeddings(
    X,
    sparse_feature_defs=None,
    dense_feature_defs=None,
    embedding_layer_def=None,
):
    sparse_feature_defs = sparse_feature_defs or []
    dense_feature_defs = dense_feature_defs or []
    # embeddings part
    if not embedding_layer_def:
        sparse_embeddings = []
    else:
        sparse_embeddings = [
            embedding_layer_def[feat.name](
                X[:, feat.column_idx : feat.column_idx + 1].long()
            )
            for feat in sparse_feature_defs
        ]
    # dense inputs part
    dense_values = [
        X[:, feat.column_idx : feat.column_idx + feat.dimension]
        for feat in dense_feature_defs
    ]
    return dense_values, sparse_embeddings


def concatenate(inputs, axis=-1):
    if len(inputs) == 1:
        return inputs[0]
    else:
        return torch.cat(inputs, dim=axis)

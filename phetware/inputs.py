from collections import OrderedDict, namedtuple

import torch
import torch.nn as nn

from phetware.layer.utils import concat_func

DEFAULT_GROUP_NAME = "default_group"


class SparseFeat(namedtuple('SparseFeat',
                            ['name', 'vocabulary_size', 'embedding_dim', 'use_hash', 'dtype', 'embedding_name',
                             'group_name', 'feat_type'])):
    __slots__ = ()

    def __new__(cls, name, vocabulary_size, embedding_dim=4, use_hash=False, dtype="int32", embedding_name=None,
                group_name=DEFAULT_GROUP_NAME, feat_type="SparseFeat"):
        if embedding_name is None:
            embedding_name = name
        if embedding_dim == "auto":
            embedding_dim = 6 * int(pow(vocabulary_size, 0.25))
        if use_hash:
            print(
                "Notice! Feature Hashing on the fly currently is not supported in torch version,you can use tensorflow version!")
        return super(SparseFeat, cls).__new__(cls, name, int(vocabulary_size), int(embedding_dim), use_hash, dtype,
                                              embedding_name, group_name, feat_type)

    def __hash__(self):
        return self.name.__hash__()

class DenseFeat(namedtuple('DenseFeat', ['name', 'dimension', 'dtype', 'feat_type'])):
    __slots__ = ()

    def __new__(cls, name, dimension=1, dtype="float32", feat_type="DenseFeat"):
        return super(DenseFeat, cls).__new__(cls, name, int(dimension), dtype, feat_type)

    def __hash__(self):
        return self.name.__hash__()


def reformat_input_features(feature_defs):
    new_feature_defs = list()
    for feat in feature_defs:
        feat_variant = globals()[feat["feat_type"]]
        new_feature_defs.append(feat_variant(**feat))
    return new_feature_defs


def build_feature_named_index_mapping(feature_defs):
    features = OrderedDict()

    start = 0
    for feat in feature_defs:
        feat_name = feat.name
        if feat_name in features:
            continue
        if isinstance(feat, SparseFeat):
            features[feat_name] = (start, start + 1)
            start += 1
        elif isinstance(feat, DenseFeat):
            features[feat_name] = (start, start + feat.dimension)
            start += feat.dimension
        else:
            raise TypeError("Invalid feature column type, got", type(feat))
    return features


def concat_dnn_inputs(sparse_embedding_list, dense_value_list):
    if len(sparse_embedding_list) > 0 and len(dense_value_list) > 0:
        sparse_dnn_input = torch.flatten(
            torch.cat(sparse_embedding_list, dim=-1), start_dim=1)
        dense_dnn_input = torch.flatten(
            torch.cat(dense_value_list, dim=-1), start_dim=1)
        return concat_func([sparse_dnn_input, dense_dnn_input])
    elif len(sparse_embedding_list) > 0:
        return torch.flatten(torch.cat(sparse_embedding_list, dim=-1), start_dim=1)
    elif len(dense_value_list) > 0:
        return torch.flatten(torch.cat(dense_value_list, dim=-1), start_dim=1)
    else:
        raise NotImplementedError


def embedding_dict_gen(sparse_feature_defs, init_std=0.0001, linear=False, sparse=False, device='cpu'):
    embedding_dict = nn.ModuleDict({
        feat.embedding_name: nn.Embedding(
            feat.vocabulary_size,
            feat.embedding_dim if not linear else 1,
            sparse=sparse) for feat in sparse_feature_defs
    })

    for tensor in embedding_dict.values():
        nn.init.normal_(tensor.weight, mean=0, std=init_std)

    return embedding_dict.to(device)


def compute_inputs_dim(
    sparse_feature_defs, dense_feature_defs, include_sparse=True,
    include_dense=True, feature_group=False
):
    input_dim = 0
    dense_input_dim = sum(
        map(lambda x: x.dimension, dense_feature_defs))
    if feature_group:
        sparse_input_dim = len(sparse_feature_defs)
    else:
        sparse_input_dim = sum(feat.embedding_dim for feat in sparse_feature_defs)
    
    if include_sparse:
        input_dim += sparse_input_dim
    if include_dense:
        input_dim += dense_input_dim
    return input_dim


def collect_inputs_and_embeddings(
    X, sparse_feature_defs, dense_feature_defs, feature_name_to_index,
    embedding_layer_def=None
):
    # embeddings part
    if not embedding_layer_def:
        sparse_embeddings = []
    else:
        sparse_embeddings = [embedding_layer_def[feat.embedding_name](X[
            :, feature_name_to_index[feat.name][0]: feature_name_to_index[feat.name][1]
        ].long()) for feat in sparse_feature_defs]
    # dense inputs part
    dense_values = [X[
        :, feature_name_to_index[feat.name][0]: feature_name_to_index[feat.name][1]
    ] for feat in dense_feature_defs]
    return dense_values, sparse_embeddings

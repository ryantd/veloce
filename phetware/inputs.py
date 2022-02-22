from collections import OrderedDict, namedtuple

import torch
import torch.nn as nn

DEFAULT_GROUP_NAME = "default_group"


class SparseFeat(
    namedtuple(
        "SparseFeat",
        [
            "name",
            "vocabulary_size",
            "column_idx",
            "embedding_dim",
            "dtype",
            "group_name",
            "feat_type",
        ],
    )
):
    key = "sparse"

    def __new__(
        cls,
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
        return super(SparseFeat, cls).__new__(
            cls,
            name,
            int(vocabulary_size),
            int(column_idx),
            int(embedding_dim),
            dtype,
            group_name,
            feat_type,
        )

    def __hash__(self):
        return self.name.__hash__()


class DenseFeat(
    namedtuple("DenseFeat", ["name", "column_idx", "dimension", "dtype", "feat_type"])
):
    key = "dense"

    def __new__(
        cls, name, column_idx, *, dimension=1, dtype="float32", feat_type="DenseFeat"
    ):
        return super(DenseFeat, cls).__new__(
            cls, name, int(column_idx), int(dimension), dtype, feat_type
        )

    def __hash__(self):
        return self.name.__hash__()


def is_feature_defs(value):
    try:
        return all(["feat_type" in v for v in value])
    except:
        return False


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
    feature_name_to_index=None,
    embedding_layer_def=None,
):
    if not feature_name_to_index:
        raise ValueError("Arg feature_name_to_index should be given")
    sparse_feature_defs = sparse_feature_defs or []
    dense_feature_defs = dense_feature_defs or []
    # embeddings part
    if not embedding_layer_def:
        sparse_embeddings = []
    else:
        sparse_embeddings = [
            embedding_layer_def[feat.name](
                X[
                    :,
                    feature_name_to_index[feat.name][0] : feature_name_to_index[
                        feat.name
                    ][1],
                ].long()
            )
            for feat in sparse_feature_defs
        ]
    # dense inputs part
    dense_values = [
        X[
            :,
            feature_name_to_index[feat.name][0] : feature_name_to_index[feat.name][1],
        ]
        for feat in dense_feature_defs
    ]
    return dense_values, sparse_embeddings


def concatenate(inputs, axis=-1):
    if len(inputs) == 1:
        return inputs[0]
    else:
        return torch.cat(inputs, dim=axis)

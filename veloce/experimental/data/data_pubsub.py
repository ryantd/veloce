from typing import Union

import ray
from pyarrow.csv import ConvertOptions

from .preprocessing import fillna, LabelEncoder, MinMaxScaler


class DataPubsub(object):
    def __init__(self) -> None:
        pass


class FilePubsub(DataPubsub):
    def __init__(self, file_path, file_format: str = None) -> None:
        super().__init__()


class CSVPubsub(FilePubsub):
    def __init__(self, file_path, *, strings_can_be_null=True, rand_seed=2022, **kwargs) -> None:
        self.publisher = ray.data.read_csv(
            file_path,
            convert_options=ConvertOptions(strings_can_be_null=strings_can_be_null),
            **kwargs,
        )
        self.rand_seed = rand_seed

    def subscribe(self):
        for obj_ref in self.publisher._blocks.iter_blocks():
            table = ray.get(obj_ref)
            for index in range(self.publisher.count()):
                yield table.take([index])
    
    def __len__(self):
        return self.publisher.count()
    
    @property
    def n_blocks(self):
        return self.publisher.num_blocks()

    def set_sparse_features(
        self,
        feature_names,
        *,
        use_fillna=True,
        use_label_encoder=True,
        fillna_value="-1",
    ):
        self.sparse_feat_names = feature_names
        if use_fillna:
            self.publisher = self.publisher.map_batches(
                fillna(feature_names, fillna_value), batch_format="pyarrow",
            )
        if use_label_encoder:
            self.publisher = self.publisher.map_batches(
                LabelEncoder(feature_names), batch_format="pyarrow",
            )
        return self

    def set_dense_features(
        self,
        feature_names,
        *,
        use_fillna=True,
        use_minmax_scaler=True,
        fillna_value: Union[int, str] = 0,
    ):
        self.dense_feat_names = feature_names
        if use_fillna:
            self.publisher = self.publisher.map_batches(
                fillna(feature_names, fillna_value), batch_format="pyarrow",
            )
        if use_minmax_scaler:
            self.publisher = self.publisher.map_batches(
                MinMaxScaler(feature_names), batch_format="pyarrow",
            )
        return self

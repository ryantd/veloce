from typing import List

import pyarrow as pa

from veloce.experimental.logger import get_logger

logger = get_logger("inner")


class fillna(object):
    def __init__(self, col_selectors: List, value):
        self.col_selectors = col_selectors
        self.fill_value = value

    def __call__(self, data):
        if isinstance(data, pa.Table):
            return self._process_pyarrow_table(data)
        else:
            raise NotImplementedError

    def _process_pyarrow_table(self, table: pa.Table) -> pa.Table:
        for col in self.col_selectors:
            new_array = pa.compute.fill_null(table.column(col), self.fill_value)
            table = table.set_column(table.column_names.index(col), col, new_array)
        return table


class LabelEncoder(object):
    def __init__(self, col_selectors: List):
        self.col_selectors = col_selectors

    def __call__(self, data):
        if isinstance(data, pa.Table):
            return self._process_pyarrow_table(data)
        else:
            raise NotImplementedError

    def _process_pyarrow_table(self, table: pa.Table) -> pa.Table:
        for col in self.col_selectors:
            new_array = pa.compute.dictionary_encode(table.column(col)).combine_chunks().indices
            table = table.set_column(table.column_names.index(col), col, new_array)
        return table

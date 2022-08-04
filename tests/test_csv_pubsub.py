import pytest

from veloce.experimental.data import CSVPubsub
from veloce.experimental.environ import init, shutdown
from veloce.experimental.logger import get_logger

logger = get_logger("test")


@pytest.fixture(scope="function")
def setup_csv_pubsub():
    init(n_cpus=1)
    pubsub = CSVPubsub("examples/dataset/ctr/criteo_mini.txt")
    yield pubsub
    shutdown()


def test_csv_pubsub(setup_csv_pubsub):
    assert setup_csv_pubsub.n_blocks == 1
    assert len(setup_csv_pubsub) == 200
    setup_csv_pubsub.set_sparse_features(feature_names=[f"C{i}" for i in range(1, 27)], use_label_encoder=False)
    rows = [i for i in setup_csv_pubsub.subscribe()]
    assert len(rows) == 200
    assert str(rows[0]["C19"][0]) == "-1"
    setup_csv_pubsub.set_sparse_features(feature_names=[f"C{i}" for i in range(1, 27)], use_label_encoder=True)
    first_row = next(setup_csv_pubsub.subscribe())
    assert first_row["C1"][0].as_py() == 0
    setup_csv_pubsub.set_dense_features(feature_names=[f"I{i}" for i in range(1, 14)], fillna_value="_min", use_minmax_scaler=False)
    first_row = next(setup_csv_pubsub.subscribe())
    assert first_row["I1"][0].as_py() == 0
    setup_csv_pubsub.set_dense_features(feature_names=[f"I{i}" for i in range(1, 14)], fillna_value="_min", use_minmax_scaler=True)
    next(setup_csv_pubsub.subscribe())
    second_row = next(setup_csv_pubsub.subscribe())
    assert second_row["I2"][0].as_py() == -1
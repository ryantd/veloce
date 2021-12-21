import torch
import ray
from ray.train import Trainer
from ray.train.callbacks import JsonLoggerCallback, TBXLoggerCallback

from phetware.train_fn import Linear

SPLIT_FACTOR = 0.8
RAND_SEED = 2021


def get_datasets():
    ds: ray.data.Dataset = ray.data.read_csv('examples/data/linear_sample.csv')
    split_index = int(ds.count() * SPLIT_FACTOR)
    train_dataset, validation_dataset = \
        ds.random_shuffle(seed=RAND_SEED).split_at_indices([split_index])
    train_dataset_pipe = \
        train_dataset.repeat().random_shuffle_each_window(seed=RAND_SEED)
    validation_dataset_pipe = validation_dataset.repeat()
    datasets = {
        "train": train_dataset_pipe,
        "validation": validation_dataset_pipe
    }
    return datasets


def train_linear_dist(num_workers=2, use_gpu=False):
    datasets = get_datasets()
    trainer = Trainer("torch", num_workers=num_workers, use_gpu=use_gpu)
    trainer.start()
    results = trainer.run(
        train_func=Linear,
        dataset=datasets,
        callbacks=[JsonLoggerCallback(), TBXLoggerCallback()],
        config={
            "lr": 1e-7,
            "hidden_size": 1,
            "batch_size": 10,
            "epochs": 10,
            "torch_dataset_options": dict(
                label_column="y",
                feature_columns=["x"],
                label_column_dtype=torch.float,
                feature_column_dtypes=[torch.float])
        })
    trainer.shutdown()
    print(f"Results: {results}")


if __name__ == "__main__":
    ray.init(num_cpus=3)
    train_linear_dist()
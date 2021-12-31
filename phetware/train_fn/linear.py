import torch
import torch.nn as nn
import ray.train as train

from phetware.model.torch import OneLinear
from phetware import Epochvisor


def Linear(config):
    batch_size = config.get("batch_size")
    hidden_size = config.get("hidden_size")
    lr = config.get("lr")
    epochs = config.get("epochs")
    torch_dataset_options = config.get("torch_dataset_options")

    # dataset setup
    train_dataset_shard = train.get_dataset_shard("train")
    validation_dataset_shard = train.get_dataset_shard("validation")
    train_dataset_iterator = train_dataset_shard.iter_datasets()
    validation_dataset_iterator = validation_dataset_shard.iter_datasets()

    # model setup
    model = OneLinear(hidden_size)
    model = train.torch.prepare_model(model)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    device = train.torch.get_device()

    ep = Epochvisor(
        epochs=epochs, train_dataset_iter=train_dataset_iterator,
        validation_dataset_iter=validation_dataset_iterator,
        dataset_options=torch_dataset_options,
        batch_size=batch_size, model=model, loss_fn=loss_fn,
        optimizer=optimizer, device=device)
    
    return ep.run_epochs()

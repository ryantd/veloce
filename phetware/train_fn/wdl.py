import torch
import torch.nn as nn
import ray.train as train

from phetware.model.torch import WideAndDeep as wdl_model
from phetware.inputs import reformat_input_features
from phetware import Epochvisor


def WideAndDeep(config):
    linear_feature_columns = config.get("linear_feature_columns")
    dnn_feature_columns = config.get("dnn_feature_columns")
    torch_dataset_options = config.get("torch_dataset_options")
    epochs = config.get("epochs")
    batch_size = config.get("batch_size")

    # params setup
    linear_feature_columns = reformat_input_features(linear_feature_columns)
    dnn_feature_columns = reformat_input_features(dnn_feature_columns)

    # dataset setup
    train_dataset_shard = train.get_dataset_shard("train")
    validation_dataset_shard = train.get_dataset_shard("validation")
    train_dataset_iterator = train_dataset_shard.iter_datasets()
    validation_dataset_iterator = validation_dataset_shard.iter_datasets()

    # model setup
    device = train.torch.get_device()
    model = wdl_model(linear_feature_columns, dnn_feature_columns, task='binary', device=device)
    model = train.torch.prepare_model(model)
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters())
    checkpoint = train.load_checkpoint() or None

    ep = Epochvisor(
        epochs=epochs, train_dataset_iter=train_dataset_iterator,
        validation_dataset_iter=validation_dataset_iterator,
        dataset_options=torch_dataset_options,
        batch_size=batch_size, model=model, loss_fn=loss_fn,
        optimizer=optimizer, device=device)
    
    return ep.run_epochs(checkpoint)

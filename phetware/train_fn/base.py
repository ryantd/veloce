import torch
import torch.nn as nn
import ray.train as train

from phetware.inputs import reformat_input_features
from phetware import Epochvisor


class BaseTrainFn(object):
    def __call__(self, config):
        self.linear_feature_columns = config.get("linear_feature_columns")
        self.dnn_feature_columns = config.get("dnn_feature_columns")
        self.torch_dataset_options = config.get("torch_dataset_options")
        self.epochs = config.get("epochs")
        self.batch_size = config.get("batch_size")
        self.loss_fn = config.get("loss_fn", nn.BCELoss)
        self.optimizer = config.get("optimizer", torch.optim.Adam)
        self.checkpoint = train.load_checkpoint() or None
        self.device = train.torch.get_device()

        # params setup
        self.linear_feature_columns = reformat_input_features(self.linear_feature_columns)
        self.dnn_feature_columns = reformat_input_features(self.dnn_feature_columns)

        # dataset setup
        train_dataset_shard = train.get_dataset_shard("train")
        validation_dataset_shard = train.get_dataset_shard("validation")
        self.train_dataset_iterator = train_dataset_shard.iter_datasets()
        self.validation_dataset_iterator = validation_dataset_shard.iter_datasets()

        # loss_fn setup
        self.loss_fn = self.loss_fn()
    
    def setup_model(self, model):
        self.model = model
        self.optimizer = self.optimizer(model.parameters())
        self.setup_epv()
    
    def setup_epv(self):
        self.epv = Epochvisor(
            epochs=self.epochs, train_dataset_iter=self.train_dataset_iterator,
            validation_dataset_iter=self.validation_dataset_iterator,
            dataset_options=self.torch_dataset_options,
            batch_size=self.batch_size, model=self.model, loss_fn=self.loss_fn,
            optimizer=self.optimizer, device=self.device,
            checkpoint=self.checkpoint)
    
    def train_and_validate_epochs(self):
        return self.epv.train_and_validate_epochs()
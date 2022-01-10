import torch
import torch.nn as nn
import torchmetrics
import ray.train as train

from phetware.util import get_package_name
from phetware import Epochvisor


class BaseTrainFn(object):
    def __call__(self, config):
        self.torch_dataset_options = config.get("torch_dataset_options", None)
        self.seed = config.get("seed", 1024)
        self.epochs = config.get("epochs", 10)
        self.batch_size = config.get("batch_size", 256)
        self.loss_fn = config.get("loss_fn", nn.BCELoss())
        self.optimizer = config.get("optimizer", torch.optim.Adam)
        self.metric_fns = config.get("metric_fns", [torchmetrics.AUROC()])
        self.output_fn = config.get("output_fn", torch.sigmoid)
        self.output_fn_args = config.get("output_fn_args", None)
        self.log_nn_arch = config.get("log_nn_arch", False)
        self.init_std = config.get("init_std", 0.0001)
        self.checkpoint = train.load_checkpoint() or None
        self.device = train.torch.get_device()

        # dataset setup
        train_dataset_shard = train.get_dataset_shard("train")
        self.train_dataset_iterator = train_dataset_shard.iter_datasets()
        try:
            validation_dataset_shard = train.get_dataset_shard("validation")
            self.validation_dataset_iterator = validation_dataset_shard.iter_datasets()
        except:
            self.validation_dataset_iterator = None
        try:
            test_dataset_shard = train.get_dataset_shard("test")
            self.test_dataset_iterator = test_dataset_shard.iter_datasets()
        except:
            self.test_dataset_iterator = None
    
    def setup_model(self, model):
        self.model = model
        if get_package_name(self.optimizer) == "torch":
            self.optimizer = self.optimizer(model.parameters())
        elif type(self.optimizer).__name__ == "OptimizerStack":
            self.optimizer.compile(self.model.module)
        else:
            raise ValueError("optimizer must be given and valid")
        self.setup_epv()
        if self.log_nn_arch: print(model)
    
    def setup_epv(self):
        self.epv = Epochvisor(
            epochs=self.epochs, train_dataset_iter=self.train_dataset_iterator,
            validation_dataset_iter=self.validation_dataset_iterator,
            test_dataset_iter=self.test_dataset_iterator,
            dataset_options=self.torch_dataset_options,
            batch_size=self.batch_size, model=self.model, loss_fn=self.loss_fn,
            optimizer=self.optimizer, metric_fns=self.metric_fns,
            device=self.device, checkpoint=self.checkpoint)
    
    def run_epochs(self):
        return self.epv.run_epochs()

import inspect

import torch
import torch.nn as nn
import torchmetrics
import ray.train as train

from phetware.util import get_package_name
from phetware.inputs import reformat_input_features, is_feature_defs
from phetware.model.torch import (
    WideAndDeep as _WideAndDeep,
    DeepFM as _DeepFM,
    PNN as _PNN,
    FNN as _FNN,
)
from phetware import Epochvisor


class BaseTrainFn(object):
    def __call__(self, config):
        self.torch_dataset_options = config.get("torch_dataset_options", None)
        self.seed = config.get("seed", 1024)
        self.epochs = config.get("epochs", 10)
        self.batch_size = config.get("batch_size", 256)
        self.loss_fn = config.get("loss_fn", nn.BCELoss())
        self.optimizer = config.get("optimizer", torch.optim.Adam)
        self.optimizer_args = config.get("optimizer_args", None) or {}
        self.metric_fns = config.get("metric_fns", [torchmetrics.AUROC()])
        self.output_fn = config.get("output_fn", torch.sigmoid)
        self.output_fn_args = config.get("output_fn_args", None)
        self.summary_nn_arch = config.get("summary_nn_arch", False)
        self.init_std = config.get("init_std", 0.0001)
        self.ddp_options = config.get("ddp_options", None) or {}
        self.use_static_graph = config.get("use_static_graph", False)
        self.checkpoint = train.load_checkpoint() or None
        self.device = train.torch.get_device()
        self.use_early_stopping = config.get("use_early_stopping", False)
        self.early_stopping_args = config.get("early_stopping_args", None) or {}

        # dataset setup
        train_dataset_shard = train.get_dataset_shard("train")
        self.train_dataset_iterator = train_dataset_shard.iter_datasets()
        try:
            validation_dataset_shard = train.get_dataset_shard("validation")
            self.validation_dataset_iterator = validation_dataset_shard.iter_datasets()
        except:
            self.validation_dataset_iterator = None

    def setup_model(self, model):
        self.model = model
        if self.use_static_graph:
            self.model._set_static_graph()
        if get_package_name(self.optimizer) == "torch" or (
            get_package_name(self.optimizer) == "phetware"
            and type(self.optimizer).__name__ != "OptimizerStack"
        ):
            self.optimizer = self.optimizer(model.parameters(), **self.optimizer_args)
        elif type(self.optimizer).__name__ == "OptimizerStack":
            self.optimizer.compile(self.model.module)
        else:
            raise ValueError("optimizer must be given and valid")
        self.setup_epv()
        if self.summary_nn_arch:
            print(model)

    def setup_epv(self):
        self.epv = Epochvisor(
            epochs=self.epochs,
            train_dataset_iter=self.train_dataset_iterator,
            validation_dataset_iter=self.validation_dataset_iterator,
            dataset_options=self.torch_dataset_options,
            batch_size=self.batch_size,
            model=self.model,
            loss_fn=self.loss_fn,
            optimizer=self.optimizer,
            metric_fns=self.metric_fns,
            device=self.device,
            checkpoint=self.checkpoint,
            use_early_stopping=self.use_early_stopping,
            early_stopping_args=self.early_stopping_args,
        )

    def run_epochs(self):
        return self.epv.run_epochs()


class Generic(BaseTrainFn):
    def __init__(self, model):
        self.model = model

    def __call__(self, origin_config):
        super(Generic, self).__call__(origin_config)
        config = dict()
        requires_arg = list(inspect.signature(self.model.__init__).parameters.keys())
        requires_arg.pop(0)
        for k, v in origin_config.items():
            if is_feature_defs(v):
                v = reformat_input_features(v)
            if k in requires_arg:
                config[k] = v
        model = train.torch.prepare_model(
            model=self.model(**config), ddp_kwargs=self.ddp_options
        )
        self.setup_model(model=model)
        return self.run_epochs()


# native train fns
WideAndDeep = Generic(_WideAndDeep)
DeepFM = Generic(_DeepFM)
PNN = Generic(_PNN)
FNN = Generic(_FNN)

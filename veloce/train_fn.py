import torch
import torch.nn as nn
import ray.train as train

from veloce.util import get_package_name, inspect_func_args
from veloce.inputs import rebuild_feature_values, find_feature_values
from veloce.model.ctr import (
    WideAndDeep as _WideAndDeep,
    DeepFM as _DeepFM,
    PNN as _PNN,
    FNN as _FNN,
)
from veloce import Epochvisor


class BaseTrainFn(object):
    def __call__(self, config):
        self.set_config(config)
        # dataset setup
        train_dataset_shard = train.get_dataset_shard("train")
        self.train_dataset_iterator = train_dataset_shard.iter_datasets()
        try:
            validation_dataset_shard = (
                self.shared_validation_dataset_shard
                or train.get_dataset_shard("validation")
            )
            self.validation_dataset_iterator = validation_dataset_shard.iter_datasets()
        except:
            self.validation_dataset_iterator = None

    def set_config(self, config):
        # basic configs setup
        self.dataset_options = config.get("dataset_options", None)
        self.seed = config.get("seed", 1024)
        self.epochs = config.get("epochs", 10)
        self.batch_size = config.get("batch_size", 256)
        self.loss_fn = config.get("loss_fn", nn.BCELoss())
        self.optimizer = config.get("optimizer", torch.optim.Adam)
        self.optimizer_args = config.get("optimizer_args", None) or {}
        self.output_fn = config.get("output_fn", torch.sigmoid)
        self.output_fn_args = config.get("output_fn_args", None)
        self.summary_nn_arch = config.get("summary_nn_arch", False)
        self.init_std = config.get("init_std", 0.0001)
        self.ddp_options = config.get("ddp_options", None) or {}
        self.use_static_graph = config.get("use_static_graph", False)
        self.use_early_stopping = config.get("use_early_stopping", False)
        self.early_stopping_args = config.get("early_stopping_args", None) or {}
        self.shared_validation_dataset_shard = config.get(
            "shared_validation_dataset", None
        )
        try:
            import torchmetrics

            self.metric_fns = config.get("metric_fns", [torchmetrics.AUROC()])
        except:
            self.metric_fns = config.get("metric_fns", None)
            if not self.metric_fns:
                raise ValueError(
                    "Arg metric_fns must be given and valid, you may use sklearn or torchmetrics metric functions"
                )
        try:
            self.checkpoint = train.load_checkpoint() or None
            self.device = train.torch.get_device()
        except:
            self.checkpoint = None
            self.device = config.get("device", "cpu")

    def setup_model(
        self, model_args, model=None, enable_optimizer=True, enable_ddp=True
    ):
        # model setup
        if (not hasattr(self, "model") or self.model is None) and model is not None:
            self.model = model
        if enable_ddp:
            self.model = train.torch.prepare_model(
                model=self.model(**model_args), ddp_kwargs=self.ddp_options
            )
        else:
            self.model = self.model(**model_args)
        if self.use_static_graph:
            self.model._set_static_graph()
        if self.summary_nn_arch:
            print(self.model)

        if enable_optimizer:
            # optimizer setup
            if get_package_name(self.optimizer) == "torch" or (
                get_package_name(self.optimizer) == "veloce"
                and type(self.optimizer).__name__ != "OptimizerStack"
            ):
                self.optimizer = self.optimizer(
                    self.model.parameters(), **self.optimizer_args
                )
            elif type(self.optimizer).__name__ == "OptimizerStack":
                self.optimizer.compile(self.model.module if enable_ddp else self.model)
            else:
                raise ValueError("Arg optimizer must be given and valid")

    def setup_epv(self):
        self.epv = Epochvisor(
            epochs=self.epochs,
            train_dataset_iter=self.train_dataset_iterator,
            validation_dataset_iter=self.validation_dataset_iterator,
            dataset_options=self.dataset_options,
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


class RecommendationFn(BaseTrainFn):
    def __init__(self, model):
        self.model = model

    def __call__(self, config):
        super(RecommendationFn, self).__call__(config)
        for k in config.keys():
            if find_feature_values(config[k]):
                config[k] = rebuild_feature_values(config[k])
        model_args = inspect_func_args(config, self.model)
        self.setup_model(model_args=model_args)
        self.setup_epv()
        return self.run_epochs()


# native train fns
WideAndDeep = RecommendationFn(_WideAndDeep)
DeepFM = RecommendationFn(_DeepFM)
PNN = RecommendationFn(_PNN)
FNN = RecommendationFn(_FNN)

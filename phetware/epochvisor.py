import time
from collections import defaultdict

import torch
import ray.train as train

from .util import get_package_name, merge_results, get_type, get_func_name


class LossMetricAccumulator(object):
    def __init__(self, loss_fn, metric_fns):
        self.sklearn_intermediates = defaultdict(int)
        self.torchm_intermediates = defaultdict(int)
        self.loss = 0
        self.loss_fn = loss_fn
        self.metric_fns = metric_fns

    def acc(self, pred, y):
        self.loss += self.loss_fn(pred, y).item()
        for fn in self.metric_fns:
            if get_package_name(fn) == "torchmetrics":
                r = fn(pred, y.int())
                if get_type(fn) != "function":
                    continue
                self.torchm_intermediates[get_func_name(fn)] += r
            elif get_package_name(fn) == "sklearn":
                self.sklearn_intermediates[get_func_name(fn)] += fn(
                    y.cpu().data.numpy(), pred.cpu().data.numpy()
                )

    def gen_result(self, num_batches, with_latest_loss=False):
        results = {get_type(self.loss_fn): self.loss / num_batches}
        # torchmetrics module compute and reset
        for fn in self.metric_fns:
            if get_package_name(fn) == "torchmetrics" and get_type(fn) != "function":
                results[get_type(fn)] = fn.compute().item()
                fn.reset()
        # torchmetrics functional compute
        for name in self.torchm_intermediates.keys():
            self.torchm_intermediates[name] /= num_batches
        # sklearn compute
        for name in self.sklearn_intermediates.keys():
            self.sklearn_intermediates[name] /= num_batches
        results.update(self.torchm_intermediates)
        results.update(self.sklearn_intermediates)

        if with_latest_loss:
            return results, self.loss / num_batches
        return results


class Epochvisor(object):
    def __init__(
        self,
        epochs,
        *,
        dataset_options,
        batch_size,
        model,
        loss_fn,
        optimizer,
        device,
        train_dataset_iter,
        checkpoint=None,
        metric_fns=None,
        validation_dataset_iter=None,
        use_early_stopping=False,
        early_stopping_args=None,
    ):
        self.epochs = epochs
        self.train_dataset_iter = train_dataset_iter
        self.validation_dataset_iter = validation_dataset_iter
        self.dataset_options = dataset_options
        self.batch_size = batch_size
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.metric_fns = metric_fns
        self.device = device
        self.checkpoint = checkpoint
        self.use_early_stopping = use_early_stopping
        self.early_stopping_args = early_stopping_args or {"patience": 2}

        if not self.metric_fns:
            raise ValueError("metric_fns must be given")

    def run_epochs(self):
        label_column = self.dataset_options["label_column"]
        feature_columns = self.dataset_options["feature_columns"]
        label_column_dtype = self.dataset_options["label_column_dtype"]
        feature_column_dtypes = self.dataset_options["feature_column_dtypes"]
        start_epoch = 0
        results = []
        is_checkpoint_based = False

        # early stopping related
        previous_epoch_loss = 1
        es_trigger_times = 0
        is_early_stopped = False

        if self.checkpoint:
            is_checkpoint_based = True
            model_state_dict = self.checkpoint.get("model_state_dict", None)
            optimizer_state_dict = self.checkpoint.get("optimizer_state_dict", None)
            start_epoch = self.checkpoint.get("epoch_id", 0)

            self.model.load_state_dict(model_state_dict)
            self.optimizer.load_state_dict(optimizer_state_dict)

        for epoch_id in range(start_epoch, self.epochs):
            start_ts = time.time()
            train_dataset = next(self.train_dataset_iter)
            train_torch_dataset = train_dataset.to_torch(
                label_column=label_column,
                feature_columns=feature_columns,
                label_column_dtype=label_column_dtype,
                feature_column_dtypes=feature_column_dtypes,
                batch_size=self.batch_size,
            )
            train_result = self.train_epoch(train_torch_dataset)
            if self.validation_dataset_iter:
                validation_dataset = next(self.validation_dataset_iter)
                validation_torch_dataset = validation_dataset.to_torch(
                    label_column=label_column,
                    feature_columns=feature_columns,
                    label_column_dtype=label_column_dtype,
                    feature_column_dtypes=feature_column_dtypes,
                    batch_size=self.batch_size,
                )
                validation_result, latest_epoch_loss = self.validate_epoch(
                    validation_torch_dataset, with_latest_loss=True
                )
            end_ts = time.time()

            # early stopping check
            if self.use_early_stopping:
                if latest_epoch_loss > previous_epoch_loss:
                    es_trigger_times += 1
                    if (
                        es_trigger_times >= self.early_stopping_args["patience"]
                        and epoch_id != self.epochs - 1
                    ):
                        is_early_stopped = True
                else:
                    es_trigger_times = 0

            # post epoch operations
            result = merge_results(
                validation_result=validation_result,
                train_result=train_result,
                time_diff=end_ts - start_ts,
                is_early_stopped=is_early_stopped,
                is_checkpoint_based=is_checkpoint_based,
            )
            train.save_checkpoint(
                epoch_id=epoch_id,
                model_state_dict=self.model.state_dict(),
                optimizer_state_dict=self.optimizer.state_dict(),
            )
            train.report(**result)
            results.append(result)
            previous_epoch_loss = latest_epoch_loss
        return results

    def train_epoch(self, train_iterable_ds):
        self.model.train()
        num_batches = 0
        lma = LossMetricAccumulator(loss_fn=self.loss_fn, metric_fns=self.metric_fns)
        for X, y in train_iterable_ds:
            X = X.to(self.device)
            y = y.to(self.device)
            pred = self.model(X)
            if hasattr(self.model.module, "get_regularization_loss"):
                loss = (
                    self.loss_fn(pred, y) + self.model.module.get_regularization_loss()
                )
            else:
                loss = self.loss_fn(pred, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # metrics part
            num_batches += 1
            lma.acc(pred=pred, y=y)
        return lma.gen_result(num_batches=num_batches)

    def validate_epoch(self, validation_iterable_ds, with_latest_loss=False):
        self.model.eval()
        num_batches = 0
        lma = LossMetricAccumulator(loss_fn=self.loss_fn, metric_fns=self.metric_fns)
        with torch.no_grad():
            for X, y in validation_iterable_ds:
                X = X.to(self.device)
                y = y.to(self.device)
                pred = self.model(X)
                # metrics part
                num_batches += 1
                lma.acc(pred=pred, y=y)
        return lma.gen_result(
            num_batches=num_batches, with_latest_loss=with_latest_loss
        )

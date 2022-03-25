import time
from types import MethodType

import torch
import ray
import numpy as np

from veloce.train_fn import BaseTrainFn
from veloce.inputs import rebuild_feature_values, find_feature_values
from veloce.util import inspect_func_args, ticks, merge_results, get_batches
from veloce.preprocessing import gen_dataset_shards
from veloce.epochvisor import LossMetricAccumulator


# patch funcs for nn modules
def get_weights(self):
    return {k: v.cpu() for k, v in self.state_dict().items()}


def set_weights(self, weights):
    self.load_state_dict(weights)


def get_gradients(self):
    grads = []
    for p in self.parameters():
        grad = None if p.grad is None else p.grad.data.cpu().numpy()
        grads.append(grad)
    return grads


def set_gradients(self, gradients):
    for g, p in zip(gradients, self.parameters()):
        if g is not None:
            p.grad = torch.from_numpy(g)


@ray.remote
class ParameterServer(BaseTrainFn):
    def __init__(self, model, config):
        self.set_config(config)
        for k in config.keys():
            if find_feature_values(config[k]):
                config[k] = rebuild_feature_values(config[k])
        model_args = inspect_func_args(config, model)
        self.setup_model(model_args=model_args, model=model, enable_ddp=False)
        # patch weight & grad funcs
        self.model.get_weights = MethodType(get_weights, self.model)
        self.model.set_gradients = MethodType(set_gradients, self.model)

    def apply_gradients(self, *gradients):
        summed_gradients = [
            np.stack(gradient_zip).sum(axis=0) for gradient_zip in zip(*gradients)
        ]
        self.optimizer.zero_grad()
        self.model.set_gradients(summed_gradients)
        self.optimizer.step()
        return self.model.get_weights()

    def get_weights(self):
        return self.model.get_weights()


@ray.remote
class DataWorker(BaseTrainFn):
    def __init__(self, model, config, n_batches):
        self.n_batches = n_batches
        self.set_config(config)
        for k in config.keys():
            if find_feature_values(config[k]):
                config[k] = rebuild_feature_values(config[k])
        model_args = inspect_func_args(config, model)
        self.setup_model(
            model_args=model_args, model=model, enable_optimizer=False, enable_ddp=False
        )
        # patch weight & grad funcs
        self.model.set_weights = MethodType(set_weights, self.model)
        self.model.get_gradients = MethodType(get_gradients, self.model)

    def consume_dataset_shard(self, dataset_shard):
        train_dataset_shard = dataset_shard["train"]
        self.train_dataset_iter = train_dataset_shard.iter_datasets()
        try:
            validation_dataset_shard = dataset_shard["validation"]
            self.validation_dataset_iter = validation_dataset_shard.iter_datasets()
        except:
            self.validation_dataset_iter = None

    def next_epoch(self, key):
        # set up torch dataset iter
        dataset_iter = iter(
            next(getattr(self, f"{key}_dataset_iter")).to_torch(
                label_column=self.dataset_options["label_column"],
                feature_columns=self.dataset_options["feature_columns"],
                label_column_dtype=self.dataset_options["label_column_dtype"],
                feature_column_dtypes=self.dataset_options["feature_column_dtypes"],
                batch_size=self.batch_size,
            )
        )
        setattr(self, f"{key}_torch_dataset_iter", dataset_iter)
        # set up LossMetricAccumulator
        self.lma = LossMetricAccumulator(
            loss_fn=self.loss_fn, metric_fns=self.metric_fns
        )

    def compute_gradients(self, weights):
        try:
            X, y = next(self.train_torch_dataset_iter)
        except StopIteration:
            self.next_epoch("train")
            X, y = next(self.train_torch_dataset_iter)
        self.model.set_weights(weights)
        self.model.train()
        X = X.to(self.device)
        y = y.to(self.device)
        self.model.zero_grad()
        pred = self.model(X)
        loss = self.loss_fn(pred, y)
        loss.backward()
        self.lma.acc(pred=pred, y=y)
        return self.model.get_gradients()

    def evaluate(self, weights):
        try:
            X, y = next(self.validation_torch_dataset_iter)
        except StopIteration:
            self.next_epoch("validation")
            X, y = next(self.validation_torch_dataset_iter)
        self.model.set_weights(weights)
        self.model.eval()
        with torch.no_grad():
            X = X.to(self.device)
            y = y.to(self.device)
            pred = self.model(X)
            self.lma.acc(pred=pred, y=y)

    def gen_result(self):
        return self.lma.gen_result(num_batches=self.n_batches)


class PSTrainer(object):
    def __init__(self, num_workers, use_gpu):
        self.num_workers = num_workers
        self.use_gpu = use_gpu

    def run(self, model, config, dataset):
        result_dict = {i: [] for i in range(self.num_workers)}
        n_train_batches, n_valid_batches = get_batches(config, self.num_workers)
        iteration = config["epochs"] * n_train_batches
        # TODO: add checkpoint and callback support
        ps = ParameterServer.remote(model, config)
        workers = [
            DataWorker.remote(model, config, n_train_batches)
            for _ in range(self.num_workers)
        ]
        dataset_shards = gen_dataset_shards(
            dataset, n_shards=self.num_workers, locality_hints=workers
        )
        ticks(
            worker.consume_dataset_shard.remote(shard)
            for shard, worker in zip(dataset_shards, workers)
        )
        ticks(worker.next_epoch.remote("train") for worker in workers)
        current_weights = ps.get_weights.remote()
        start_ts = time.time()
        for i in range(iteration):
            # train epoch
            gradients = [
                worker.compute_gradients.remote(current_weights) for worker in workers
            ]
            current_weights = ray.get(ps.apply_gradients.remote(*gradients))
            if i % n_train_batches == 0:
                train_results = ray.get(
                    [worker.gen_result.remote() for worker in workers]
                )
                ticks(worker.next_epoch.remote("validation") for worker in workers)
                # validation epoch
                for _ in range(n_valid_batches):
                    ray.get(
                        [worker.evaluate.remote(current_weights) for worker in workers]
                    )
                validation_results = ray.get(
                    [worker.gen_result.remote() for worker in workers]
                )
                end_ts = time.time()
                for j in range(self.num_workers):
                    result_dict[j].append(
                        merge_results(
                            validation_result=validation_results[j],
                            train_result=train_results[j],
                            time_diff=end_ts - start_ts,
                        )
                    )
                start_ts = time.time()
        return result_dict.values()

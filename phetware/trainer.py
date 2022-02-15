import copy

from ray.train import Trainer
from ray.train.callbacks import JsonLoggerCallback, TBXLoggerCallback

from phetware.train_fn import Generic
from phetware.callback import EarlyStoppingCallback

callback_mapping = {"json": JsonLoggerCallback, "tbx": TBXLoggerCallback}
DefaultRun = {}


class NeuralNetTrainer(object):
    def __init__(
        self,
        module,
        module_params,
        dataset,
        dataset_options,
        batch_size,
        loss_fn,
        optimizer,
        metric_fns,
        num_workers,
        use_gpu,
        epochs=0,
        use_static_graph=False,
        optimizer_args=None,
        ddp_options=None,
        callbacks=None,
        use_early_stopping=False,
        early_stopping_args=None,
    ):
        self.model = module
        self.module_params = module_params
        self.ddp_options = ddp_options
        self.dataset = dataset
        self.dataset_options = dataset_options

        self.epochs = epochs
        self.batch_size = batch_size
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.optimizer_args = optimizer_args or {}
        self.metric_fns = metric_fns
        self.use_static_graph = use_static_graph
        self.use_early_stopping = use_early_stopping
        self.early_stopping_args = early_stopping_args

        self.callbacks = callbacks
        self.num_workers = num_workers
        self.use_gpu = use_gpu

        if self.callbacks is not None:
            self.callbacks = [callback_mapping[l]() for l in self.callbacks]
        else:
            self.callbacks = []

    def run(self, multi_runs=None, runs_mode="sync", use_checkpoint=False):
        if runs_mode != "sync":
            raise ValueError("Arg run_mode not support non-sync")

        if multi_runs is None:
            self.multi_runs = [DefaultRun]
        else:
            self.multi_runs = multi_runs
            use_checkpoint = True
        results = []
        latest_ckpt = None

        trainer = Trainer("torch", num_workers=self.num_workers, use_gpu=self.use_gpu)
        trainer.start()
        for addons in self.multi_runs:
            run_configs = dict(
                epochs=self.epochs,
                batch_size=self.batch_size,
                loss_fn=self.loss_fn,
                optimizer=self.optimizer,
                optimizer_args=self.optimizer_args,
                metric_fns=self.metric_fns,
                ddp_options=self.ddp_options,
                torch_dataset_options=self.dataset_options,
                use_static_graph=self.use_static_graph,
                use_early_stopping=self.use_early_stopping,
                early_stopping_args=self.early_stopping_args,
                **self.module_params
            )
            try:
                addons_module_params = addons.pop("module_params")
            except:
                addons_module_params = {}
            run_configs.update(addons)
            run_configs.update(addons_module_params)

            # early stopping part
            if not self.use_early_stopping:
                results.append(
                    trainer.run(
                        train_func=Generic(self.model),
                        dataset=self.dataset,
                        callbacks=self.callbacks,
                        config=run_configs,
                        checkpoint=latest_ckpt,
                    )
                )

            else:
                # hacky solution on EarlyStopping
                _es = EarlyStoppingCallback(trainer)
                _callbacks = copy.deepcopy(self.callbacks)
                _callbacks.append(_es)
                try:
                    trainer.run(
                        train_func=Generic(self.model),
                        dataset=self.dataset,
                        callbacks=_callbacks,
                        config=run_configs,
                        checkpoint=latest_ckpt,
                    )
                except:
                    pass
                results.append(_es.stored_results)

            # checkpoint part
            if use_checkpoint:
                latest_ckpt = trainer.latest_checkpoint
                if latest_ckpt:
                    latest_ckpt["epoch_id"] = 0

        if not self.use_early_stopping:
            trainer.shutdown()
        return results

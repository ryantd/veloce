import copy

from ray.train import Trainer
from ray.train.callbacks import JsonLoggerCallback, TBXLoggerCallback

from phetware.train_fn import Generic

callback_mapping = {"json": JsonLoggerCallback, "tbx": TBXLoggerCallback}
DefaultRun = {}


class NeuralNetTrainer(object):
    def __init__(
        self,
        module,
        module_params,
        dataset,
        dataset_options,
        epochs,
        batch_size,
        loss_fn,
        optimizer,
        metric_fns,
        num_workers,
        use_gpu,
        ddp_options=None,
        callbacks=None,
    ):
        self.model = module
        self.model_params = module_params
        self.ddp_options = ddp_options
        self.dataset = dataset
        self.dataset_options = dataset_options

        self.epochs = epochs
        self.batch_size = batch_size
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.metric_fns = metric_fns

        self.callbacks = callbacks
        self.num_workers = num_workers
        self.use_gpu = use_gpu

        if self.callbacks is not None:
            self.callbacks = [callback_mapping[l]() for l in self.callbacks]
        else:
            self.callbacks = []

    def run(self, multi_runs=None, runs_mode="sync", use_checkpoint=True):
        if runs_mode != "sync":
            raise ValueError("Arg run_mode not support non-sync")

        if multi_runs is None:
            self.multi_runs = [DefaultRun]
        else:
            self.multi_runs = multi_runs
        results = []
        latest_ckpt = None

        trainer = Trainer("torch", num_workers=self.num_workers, use_gpu=self.use_gpu)
        trainer.start()
        for run_addons in self.multi_runs:
            addon_params = (
                # only support module_params update
                run_addons["module_params"]
                if "module_params" in run_addons
                else {}
            )
            local_params = copy.deepcopy(self.model_params)
            local_params.update(addon_params)
            results.append(
                trainer.run(
                    train_func=Generic(self.model),
                    dataset=self.dataset,
                    callbacks=self.callbacks,
                    config=dict(
                        epochs=self.epochs,
                        batch_size=self.batch_size,
                        loss_fn=self.loss_fn,
                        optimizer=self.optimizer,
                        metric_fns=self.metric_fns,
                        ddp_options=self.ddp_options,
                        torch_dataset_options=self.dataset_options,
                        **local_params,
                    ),
                    checkpoint=latest_ckpt,
                )
            )
            if use_checkpoint:
                latest_ckpt = trainer.latest_checkpoint
                latest_ckpt["epoch_id"] = 0
        trainer.shutdown()
        return results

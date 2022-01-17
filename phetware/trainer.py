from ray.train import Trainer
from ray.train.callbacks import JsonLoggerCallback, TBXLoggerCallback
from phetware.train_fn import Generic

callback_mapping = {"json": JsonLoggerCallback, "tbx": TBXLoggerCallback}


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
        callbacks=None,
    ):
        self.model = module
        self.model_params = module_params
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

    def run(self):
        trainer = Trainer("torch", num_workers=self.num_workers, use_gpu=self.use_gpu)
        trainer.start()
        results = trainer.run(
            train_func=Generic(self.model),
            dataset=self.dataset,
            callbacks=self.callbacks,
            config=dict(
                epochs=self.epochs,
                batch_size=self.batch_size,
                loss_fn=self.loss_fn,
                optimizer=self.optimizer,
                metric_fns=self.metric_fns,
                torch_dataset_options=self.dataset_options,
                **self.model_params
            ),
        )
        trainer.shutdown()
        return results

import torch
import ray.train as train


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
        checkpoint,
        train_dataset_iter,
        metric_fns=None,
        validation_dataset_iter=None,
        test_dataset_iter=None,
        printable_batch_interval=10,
        verbose=0,
    ):
        self.epochs = epochs
        self.train_dataset_iter = train_dataset_iter
        self.validation_dataset_iter = validation_dataset_iter
        self.test_dataset_iter = test_dataset_iter
        self.dataset_options = dataset_options
        self.batch_size = batch_size
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.metric_fns = metric_fns
        self.device = device
        self.checkpoint = checkpoint
        self.printable_batch_interval = printable_batch_interval
        self.verbose = verbose

        if not self.metric_fns:
            raise ValueError("metric_fns must be given")

    def run_epochs(self):
        label_column = self.dataset_options["label_column"]
        feature_columns = self.dataset_options["feature_columns"]
        label_column_dtype = self.dataset_options["label_column_dtype"]
        feature_column_dtypes = self.dataset_options["feature_column_dtypes"]
        start_epoch = 0

        if self.checkpoint:
            model_state_dict = self.checkpoint.get("model_state_dict", None)
            optimizer_state_dict = self.checkpoint.get("optimizer_state_dict", None)
            start_epoch = self.checkpoint.get("epoch", -1) + 1

            self.model.load_state_dict(model_state_dict)
            self.optimizer.load_state_dict(optimizer_state_dict)

        results = []
        for epoch_id in range(start_epoch, self.epochs):
            train_dataset = next(self.train_dataset_iter)
            train_torch_dataset = train_dataset.to_torch(
                label_column=label_column,
                feature_columns=feature_columns,
                label_column_dtype=label_column_dtype,
                feature_column_dtypes=feature_column_dtypes,
                batch_size=self.batch_size,
            )
            self.train_epoch(train_torch_dataset)

            if self.validation_dataset_iter:
                validation_dataset = next(self.validation_dataset_iter)
                validation_torch_dataset = validation_dataset.to_torch(
                    label_column=label_column,
                    feature_columns=feature_columns,
                    label_column_dtype=label_column_dtype,
                    feature_column_dtypes=feature_column_dtypes,
                    batch_size=self.batch_size)
                self.validate_epoch(validation_torch_dataset)
            
            if self.test_dataset_iter:
                test_dataset = next(self.test_dataset_iter)
                test_torch_dataset = test_dataset.to_torch(
                    label_column=label_column,
                    feature_columns=feature_columns,
                    label_column_dtype=label_column_dtype,
                    feature_column_dtypes=feature_column_dtypes,
                    batch_size=self.batch_size)
                result = self.test_epoch(test_torch_dataset)

            train.save_checkpoint(
                epoch=epoch_id,
                model_state_dict=self.model.state_dict(),
                optimizer_state_dict=self.optimizer.state_dict())
            train.report(**result)
            results.append(result)
        return results

    def train_epoch(self, train_iterable_ds):
        self.model.train()
        for _, (X, y) in enumerate(train_iterable_ds):
            X = X.to(self.device)
            y = y.to(self.device)

            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def validate_epoch(self, validation_iterable_ds):
        num_batches = 0
        self.model.eval()
        loss = 0
        with torch.no_grad():
            for _, (X, y) in enumerate(validation_iterable_ds):
                X = X.to(self.device)
                y = y.to(self.device)
                num_batches += 1
                pred = self.model(X)
                loss += self.loss_fn(pred, y).item()
        loss /= num_batches
        result = {"loss": loss}
        return result
    
    def test_epoch(self, test_iterable_ds):
        for _, (X, y) in enumerate(test_iterable_ds):
            X = X.to(self.device)
            y = y.to(self.device).int()
            pred = self.model(X)
            [fn(pred, y) for fn in self.metric_fns]
        result = {type(fn).__name__: fn.compute().item() for fn in self.metric_fns}
        [fn.reset() for fn in self.metric_fns]
        return result

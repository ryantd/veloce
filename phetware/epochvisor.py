import torch
import ray.train as train


class Epochvisor(object):
    def __init__(
        self,
        epochs,
        *,
        train_dataset_iter,
        validation_dataset_iter,
        dataset_options,
        batch_size,
        model,
        loss_fn,
        optimizer,
        device,
        printable_batch_interval=10,
        verbose=0,
    ):
        self.epochs = epochs
        self.train_dataset_iter = train_dataset_iter
        self.validation_dataset_iter = validation_dataset_iter
        self.dataset_options = dataset_options
        self.batch_size = batch_size
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.printable_batch_interval = printable_batch_interval
        self.verbose = verbose

    def run_epochs(self, checkpoint=None):
        label_column = self.dataset_options["label_column"]
        feature_columns = self.dataset_options["feature_columns"]
        label_column_dtype = self.dataset_options["label_column_dtype"]
        feature_column_dtypes = self.dataset_options["feature_column_dtypes"]
        start_epoch = 0

        if checkpoint:
            model_state_dict = checkpoint.get("model_state_dict", None)
            optimizer_state_dict = checkpoint.get("optimizer_state_dict", None)
            start_epoch = checkpoint.get("epoch", -1) + 1

            self.model.load_state_dict(model_state_dict)
            self.optimizer.load_state_dict(optimizer_state_dict)

        results = []
        for epoch_id in range(start_epoch, self.epochs):
            train_dataset = next(self.train_dataset_iter)
            validation_dataset = next(self.validation_dataset_iter)

            train_torch_dataset = train_dataset.to_torch(
                label_column=label_column,
                feature_columns=feature_columns,
                label_column_dtype=label_column_dtype,
                feature_column_dtypes=feature_column_dtypes,
                batch_size=self.batch_size,
            )
            validation_torch_dataset = validation_dataset.to_torch(
                label_column=label_column,
                feature_columns=feature_columns,
                label_column_dtype=label_column_dtype,
                feature_column_dtypes=feature_column_dtypes,
                batch_size=self.batch_size)

            self.train_epoch(train_torch_dataset)
            result = self.validate_epoch(validation_torch_dataset)
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

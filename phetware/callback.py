import copy

from ray.train import TrainingCallback

ignore_keys = {"_timestamp", "_time_this_iter_s", "_training_iteration"}


class EarlyStoppingCallback(TrainingCallback):
    def __init__(self, trainer):
        self.trainer = trainer
        self.stored_results = []

    def handle_result(self, results):
        for worker_id, origin_worker_result in enumerate(results):
            worker_result = copy.deepcopy(origin_worker_result)
            for key in ignore_keys:
                worker_result.pop(key)
            if len(self.stored_results) != len(results):
                self.stored_results.append([worker_result])
            else:
                self.stored_results[worker_id].append(worker_result)
            if (
                "is_early_stopped" in worker_result
                and worker_result["is_early_stopped"]
            ):
                self.trainer.shutdown()

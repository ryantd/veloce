import ray.train as train

from phetware.model.torch import DeepFM as _DeepFM
from .base import BaseTrainFn


class DeepFM(BaseTrainFn):
    def __call__(self, config):
        super(DeepFM, self).__call__(config)
        model = train.torch.prepare_model(_DeepFM(
            self.linear_feature_columns,
            self.dnn_feature_columns,
            task='binary', device=self.device))
        self.setup_model(model=model)
        return self.train_and_validate_epochs()

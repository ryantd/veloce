import ray.train as train

from phetware.model.torch import WideAndDeep as _WideAndDeep
from .base import BaseTrainFn


class WideAndDeep(BaseTrainFn):
    def __call__(self, config):
        super(WideAndDeep, self).__call__(config)
        model = train.torch.prepare_model(_WideAndDeep(
            self.linear_feature_columns,
            self.dnn_feature_columns,
            task='binary', device=self.device))
        self.setup_model(model=model)
        return self.run_epochs()

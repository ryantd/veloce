import ray.train as train

from phetware.model.torch import WideAndDeep as _WideAndDeep
from .base import BaseTrainFn


class WideAndDeep(BaseTrainFn):
    def __call__(self, config):
        super(WideAndDeep, self).__call__(config)
        self.dnn_dropout = config.get("dnn_dropout")

        model = train.torch.prepare_model(_WideAndDeep(
            self.linear_feature_columns,
            self.dnn_feature_columns,
            dnn_dropout=self.dnn_dropout,
            task='binary', device=self.device))
        self.setup_model(model=model)
        return self.run_epochs()

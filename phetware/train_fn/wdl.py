import ray.train as train

from phetware.model.torch import WideAndDeep as _WideAndDeep
from .base import BaseTrainFn


class WideAndDeep(BaseTrainFn):
    def __call__(self, config):
        super(WideAndDeep, self).__call__(config)
        self.dnn_dropout = config.get("dnn_dropout")

        model = train.torch.prepare_model(_WideAndDeep(
            self.linear_feature_columns, self.dnn_feature_columns,
            dnn_dropout=self.dnn_dropout, output_fn=self.output_fn,
            output_fn_args=self.output_fn_args, device=self.device,
            seed=self.seed))
        
        self.setup_model(model=model)
        return self.run_epochs()

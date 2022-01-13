import ray.train as train

from phetware.inputs import reformat_input_features
from .base import BaseTrainFn


class Generic(BaseTrainFn):
    def __init__(self, model):
        self.model = model

    def __call__(self, origin_config):
        super(Generic, self).__call__(origin_config)
        config = dict()
        for k, v in origin_config.items():
            if k.endswith("_feature_defs"):
                config[k] = reformat_input_features(v)
            else:
                config[k] = v
        model = train.torch.prepare_model(self.model(**config))
        self.setup_model(model=model)
        return self.run_epochs()

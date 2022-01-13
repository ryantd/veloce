import ray.train as train

from phetware.inputs import reformat_input_features
from phetware.model.torch import WideAndDeep as _WideAndDeep
from .base import BaseTrainFn


class WideAndDeep(BaseTrainFn):
    def __call__(self, config):
        super(WideAndDeep, self).__call__(config)
        self.linear_feature_defs = reformat_input_features(
            config.get("linear_feature_defs", None))
        self.dnn_feature_defs = reformat_input_features(
            config.get("dnn_feature_defs", None))

        self.dnn_hidden_units = config.get("dnn_hidden_units", (256, 128))
        self.dnn_use_bn = config.get("dnn_use_bn", False)
        self.dnn_activation = config.get("dnn_activation", "relu")
        self.dnn_dropout = config.get("dnn_dropout", 0.2)
        self.l2_reg_linear = config.get("l2_reg_linear", 1e-5)
        self.l2_reg_embedding = config.get("l2_reg_embedding", 1e-5)
        self.l2_reg_dnn = config.get("l2_reg_dnn", 0)

        model = train.torch.prepare_model(_WideAndDeep(
            linear_feature_defs=self.linear_feature_defs,
            dnn_feature_defs=self.dnn_feature_defs,
            seed=self.seed, output_fn=self.output_fn,
            output_fn_args=self.output_fn_args, device=self.device,
            dnn_hidden_units=self.dnn_hidden_units, dnn_use_bn=self.dnn_use_bn,
            dnn_activation=self.dnn_activation, dnn_dropout=self.dnn_dropout,
            l2_reg_linear=self.l2_reg_linear,
            l2_reg_embedding=self.l2_reg_embedding,
            l2_reg_dnn=self.l2_reg_dnn,
            init_std=self.init_std))
        
        self.setup_model(model=model)
        return self.run_epochs()

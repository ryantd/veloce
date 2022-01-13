import ray.train as train

from phetware.inputs import reformat_input_features
from phetware.model.torch import PNN as _PNN
from .base import BaseTrainFn


class PNN(BaseTrainFn):
    def __call__(self, config):
        super(PNN, self).__call__(config)
        self.dnn_feature_defs = reformat_input_features(
            config.get("dnn_feature_defs", None))

        self.use_inner = config.get("use_inner", True)
        self.use_outter = config.get("use_outter", False)
        self.outter_kernel_type = config.get("outter_kernel_type", "mat")
        self.dnn_hidden_units = config.get("dnn_hidden_units", (256, 128))
        self.dnn_use_bn = config.get("dnn_use_bn", False)
        self.dnn_activation = config.get("dnn_activation", "relu")
        self.dnn_dropout = config.get("dnn_dropout", 0.2)
        self.l2_reg_embedding = config.get("l2_reg_embedding", 1e-5)
        self.l2_reg_dnn = config.get("l2_reg_dnn", 0)

        model = train.torch.prepare_model(_PNN(
            dnn_feature_defs=self.dnn_feature_defs,
            use_inner=self.use_inner, use_outter=self.use_outter,
            outter_kernel_type=self.outter_kernel_type,
            seed=self.seed, output_fn=self.output_fn,
            output_fn_args=self.output_fn_args, device=self.device,
            dnn_hidden_units=self.dnn_hidden_units, dnn_use_bn=self.dnn_use_bn,
            dnn_activation=self.dnn_activation, dnn_dropout=self.dnn_dropout,
            l2_reg_embedding=self.l2_reg_embedding,
            l2_reg_dnn=self.l2_reg_dnn,
            init_std=self.init_std))
        
        self.setup_model(model=model)
        return self.run_epochs()

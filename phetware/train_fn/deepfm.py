import ray.train as train

from phetware.inputs import reformat_input_features
from phetware.model.torch import DeepFM as _DeepFM
from .base import BaseTrainFn


class DeepFM(BaseTrainFn):
    def __call__(self, config):
        super(DeepFM, self).__call__(config)
        self.fm_1_feature_defs = reformat_input_features(
            config.get("fm_1_feature_defs", None))
        self.fm_2_feature_defs = reformat_input_features(
            config.get("fm_2_feature_defs", None))
        self.dnn_feature_defs = reformat_input_features(
            config.get("dnn_feature_defs", None))

        self.use_fm = config.get("use_fm", True)
        self.dnn_hidden_units = config.get("dnn_hidden_units", (256, 128))
        self.dnn_use_bn = config.get("dnn_use_bn", False)
        self.dnn_activation = config.get("dnn_activation", "relu")
        self.dnn_dropout = config.get("dnn_dropout", 0.2)
        self.l2_reg_embedding = config.get("l2_reg_embedding", 1e-4)
        self.l2_reg_dnn = config.get("l2_reg_dnn", 0)
        self.l2_reg_fm_1 = config.get("l2_reg_fm_1", 1e-4)
        self.l2_reg_fm_2 = config.get("l2_reg_fm_2", 1e-4)

        model = train.torch.prepare_model(_DeepFM(
            self.fm_1_feature_defs,
            self.fm_2_feature_defs,
            self.dnn_feature_defs,
            use_fm=self.use_fm, seed=self.seed, output_fn=self.output_fn,
            output_fn_args=self.output_fn_args, device=self.device,
            dnn_hidden_units=self.dnn_hidden_units, dnn_use_bn=self.dnn_use_bn,
            dnn_activation=self.dnn_activation, dnn_dropout=self.dnn_dropout,
            l2_reg_fm_1=self.l2_reg_fm_1, l2_reg_fm_2=self.l2_reg_fm_2,
            l2_reg_embedding=self.l2_reg_embedding,
            l2_reg_dnn=self.l2_reg_dnn,
            init_std=self.init_std))
        
        self.setup_model(model=model)
        return self.run_epochs()

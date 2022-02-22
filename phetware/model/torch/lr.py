import torch

from phetware.layer import OutputLayer
from phetware.model.torch.base import Linear, BaseModel


class LR(BaseModel):
    def __init__(
        self,
        linear_feature_defs=None,
        l2_reg_linear=1e-3,
        seed=1024,
        device="cpu",
        output_fn=torch.sigmoid,
        output_fn_args=None,
        **kwargs
    ):
        super(LR, self).__init__(
            linear_feature_defs=linear_feature_defs,
            seed=seed,
            device=device,
        )
        self.linear = Linear(
            sparse_feature_defs=self.fds.linear_defs_sparse,
            dense_feature_defs=self.fds.linear_defs_dense,
            feature_named_index_mapping=self.feature_name_to_index,
            device=device,
        )
        self.add_regularization_weight(self.linear.parameters(), l2=l2_reg_linear)
        self.output = OutputLayer(output_fn=output_fn, output_fn_args=output_fn_args)
        self.to(device)

    def forward(self, X):
        logit = self.linear(X)
        return self.output(logit)

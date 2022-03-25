import torch

from veloce.layer import OutputLayer
from veloce.model.ctr.base import Linear, BaseModel


class LR(BaseModel):
    def __init__(
        self,
        dense_feature_defs=None,
        sparse_feature_defs=None,
        l2_reg_linear=1e-3,
        seed=1024,
        device="cpu",
        output_fn=torch.sigmoid,
        output_fn_args=None,
    ):
        super(LR, self).__init__(seed=seed, device=device)
        self.linear = Linear(
            sparse_feature_defs=sparse_feature_defs,
            dense_feature_defs=dense_feature_defs,
            device=device,
        )
        self.add_regularization_weight(self.linear.parameters(), l2=l2_reg_linear)
        self.output = OutputLayer(output_fn=output_fn, output_fn_args=output_fn_args)

    def forward(self, X):
        logit = self.linear(X)
        return self.output(logit)

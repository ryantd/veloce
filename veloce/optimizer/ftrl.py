import torch
from torch.optim.optimizer import Optimizer


class FTRL(Optimizer):
    def __init__(self, params, lr=0.1, beta=1.0, l1=0.01, weight_decay=0):
        if not 0.0 < lr:
            raise ValueError("Invalid lr parameter: {}".format(lr))
        if not 0.0 < beta:
            raise ValueError("Invalid beta parameter: {}".format(beta))
        if not 0.0 <= l1:
            raise ValueError("Invalid l1 parameter: {}".format(l1))

        defaults = dict(lr=lr, beta=beta, l1=l1, weight_decay=weight_decay)
        super(FTRL, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                if len(state) == 0:
                    state["z"] = torch.zeros_like(p.data)
                    state["n"] = torch.zeros_like(p.data)
                # update z, n
                z, n = state["z"], state["n"]
                sigma = -n.sqrt()
                n.add_(grad ** 2)
                denom = n.sqrt()
                sigma += denom
                sigma /= group["lr"]
                z.add_(grad - sigma * p.data)
                # update weight
                denom += group["beta"]
                denom /= group["lr"]
                denom += group["weight_decay"]
                d = z.sign() * torch.maximum(z.abs() - group["l1"], torch.zeros_like(z))
                p.data = -d / denom
        return loss

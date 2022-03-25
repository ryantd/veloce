from veloce.util import get_module_name


class OptimizerStack(object):
    def __init__(self, *optimizers):
        self.optimizers_stash = optimizers
        self.optimizers = dict()

    def compile(self, model_inst):
        for optim in self.optimizers_stash:
            model_layer = getattr(model_inst, optim["model_key"])
            if "args" not in optim:
                optim["args"] = dict()
            self.optimizers[get_module_name(optim["cls"])] = optim["cls"](
                model_layer.parameters(), **optim["args"]
            )

    def zero_grad(self):
        for optim in self.optimizers.values():
            optim.zero_grad()

    def step(self):
        for optim in self.optimizers.values():
            optim.step()

    def state_dict(self):
        return {k: v.state_dict() for k, v in self.optimizers.items()}

    def load_state_dict(self, state_dict):
        for key, optim in self.optimizers.items():
            optim.load_state_dict(state_dict[key])

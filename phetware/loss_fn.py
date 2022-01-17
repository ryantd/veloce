class LossFnStack(object):
    def __init__(self, *loss_fns):
        if sum([fn_wrap["weight"] for fn_wrap in loss_fns]) != 1:
            raise ValueError("The sum of all weights in LossFnStack must be 1")
        self.loss_fns = loss_fns

    def __call__(self, pred, y):
        return sum(
            [fn_wrap["weight"] * fn_wrap["fn"](pred, y) for fn_wrap in self.loss_fns]
        )

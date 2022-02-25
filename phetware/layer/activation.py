import torch.nn as nn


class Linear(nn.Module):
    def __init__(self, **kwargs):
        super(Linear, self).__init__()

    def forward(self, inputs):
        return inputs


def activation_gen(name):
    """Generate activation layers

    Args:
        name: str or nn.Module, name of activation function
    Return:
        layer: activation layer
    """
    if isinstance(name, str):
        if name.lower() == "sigmoid":
            layer = nn.Sigmoid()
        elif name.lower() == "linear":
            layer = Linear()
        elif name.lower() == "relu":
            layer = nn.ReLU(inplace=True)
        elif name.lower() == "prelu":
            layer = nn.PReLU()
    elif issubclass(name, nn.Module):
        layer = name()
    else:
        raise NotImplementedError

    return layer

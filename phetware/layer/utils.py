import torch


def concat_func(inputs, axis=-1):
    if len(inputs) == 1:
        return inputs[0]
    else:
        return torch.cat(inputs, dim=axis)
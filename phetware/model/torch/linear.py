import torch.nn as nn


class OneLinear(nn.Module):
    def __init__(self, hidden_size):
        super(OneLinear, self).__init__()
        self.linear = nn.Linear(1, hidden_size)

    def forward(self, x):
        return self.linear(x)

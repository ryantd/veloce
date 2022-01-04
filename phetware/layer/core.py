import torch
import torch.nn as nn

from .activation import activation_gen


class DNN(nn.Module):
    def __init__(
        self, inputs_dim, hidden_units, activation='relu', l2_reg=0,
        dropout_rate=0, use_bn=False, init_std=0.0001, dice_dim=3, seed=1024,
        device='cpu'
    ):
        super(DNN, self).__init__()
        if len(hidden_units) == 0:
            raise ValueError("hidden_units is empty!!")

        self.dropout = nn.Dropout(dropout_rate)
        self.seed = seed
        self.l2_reg = l2_reg
        self.use_bn = use_bn
        hidden_units = [inputs_dim] + list(hidden_units)

        # setup layers of dnn network
        self.linear_layers = nn.ModuleList()
        self.activation_layers = nn.ModuleList()
        if self.use_bn:
            self.bn_layers = nn.ModuleList()
        
        for i in range(len(hidden_units) - 1):
            self.linear_layers.append(
                nn.Linear(hidden_units[i], hidden_units[i + 1]))
            self.activation_layers.append(
                activation_gen(activation, hidden_units[i + 1], dice_dim))
            if self.use_bn:
                self.bn_layers.append(nn.BatchNorm1d(hidden_units[i + 1]))

        for name, tensor in self.linear_layers.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=init_std)

        self.to(device)

    def forward(self, inputs):
        current = inputs
        for i in range(len(self.linear_layers)):
            layer_out = self.linear_layers[i](current)
            if self.use_bn:
                layer_out = self.bn_layers[i](layer_out)
            layer_out = self.activation_layers[i](layer_out)
            layer_out = self.dropout(layer_out)
            current = layer_out
        return current


class OutputLayer(nn.Module):
    def __init__(self, task="binary", use_bias=True):
        if task not in ["binary", "multiclass", "regression"]:
            raise ValueError("task must be binary, multiclass or regression")

        super(OutputLayer, self).__init__()
        self.use_bias = use_bias
        self.task = task
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros((1,)))

    def forward(self, X):
        output = X
        if self.use_bias:
            output += self.bias
        if self.task == "binary":
            output = torch.sigmoid(output)
        return output

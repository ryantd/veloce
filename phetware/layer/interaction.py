import torch
import torch.nn as nn


class FM(nn.Module):
    def __init__(self):
        super(FM, self).__init__()

    def forward(self, inputs):
        square_of_sum = torch.pow(torch.sum(inputs, dim=1), 2)
        sum_of_square = torch.sum(inputs * inputs, dim=1)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * torch.sum(cross_term, dim=1, keepdim=True)
        return cross_term


class InnerProduct(nn.Module):
    def __init__(self, reduce_sum=True, device="cpu"):
        super(InnerProduct, self).__init__()
        self.reduce_sum = reduce_sum
        self.to(device)

    def forward(self, inputs):
        embed_list = inputs
        row = []
        col = []
        num_inputs = len(embed_list)
        for i in range(num_inputs - 1):
            for j in range(i + 1, num_inputs):
                row.append(i)
                col.append(j)
        p = torch.cat([embed_list[idx] for idx in row], dim=1)  # batch num_pairs k
        q = torch.cat([embed_list[idx] for idx in col], dim=1)
        inner_product = p * q
        if self.reduce_sum:
            inner_product = torch.sum(inner_product, dim=2, keepdim=True)
        return inner_product


class OutterProduct(nn.Module):
    def __init__(
        self, field_size, embedding_size, kernel_type="mat", seed=1024, device="cpu"
    ):
        super(OutterProduct, self).__init__()
        self.kernel_type = kernel_type

        num_inputs = field_size
        num_pairs = int(num_inputs * (num_inputs - 1) / 2)
        embed_size = embedding_size
        if self.kernel_type == "mat":

            self.kernel = nn.Parameter(torch.Tensor(embed_size, num_pairs, embed_size))

        elif self.kernel_type == "vec":
            self.kernel = nn.Parameter(torch.Tensor(num_pairs, embed_size))

        elif self.kernel_type == "num":
            self.kernel = nn.Parameter(torch.Tensor(num_pairs, 1))
        nn.init.xavier_uniform_(self.kernel)
        self.to(device)

    def forward(self, inputs):
        embed_list = inputs
        row = []
        col = []
        num_inputs = len(embed_list)
        for i in range(num_inputs - 1):
            for j in range(i + 1, num_inputs):
                row.append(i)
                col.append(j)
        p = torch.cat([embed_list[idx] for idx in row], dim=1)  # batch num_pairs k
        q = torch.cat([embed_list[idx] for idx in col], dim=1)

        if self.kernel_type == "mat":
            p.unsqueeze_(dim=1)
            kp = torch.sum(
                torch.mul(
                    torch.transpose(torch.sum(torch.mul(p, self.kernel), dim=-1), 2, 1),
                    q,
                ),
                dim=-1,
            )
        else:
            k = torch.unsqueeze(self.kernel, 0)
            kp = torch.sum(p * q * k, dim=-1)
        return kp

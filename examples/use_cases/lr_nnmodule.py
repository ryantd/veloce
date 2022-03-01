import torch
import torch.nn as nn
from torchmetrics.functional import auroc

from phetware.util import pprint_results
from phetware.environ import environ_validate
from phetware.optimizer import FTRL
from phetware import NeuralNetTrainer
from phetware.preprocessing import DataLoader

"""
A LR use-case of phetware

What we have in the file:
    1. A Native PyTorch nn.Module
    2. Use phetware's DataLoader to load dataset, define features and do
    preprocessing. This component is leveraged by _Ray Data_.
    3. Use NeuralNetTrainer to launch a data scientist-friendly training
    lifecycle. This component is leveraged by _Ray Train_.
"""


class LR(nn.Module):
    def __init__(
        self,
        dense_defs,
        sparse_defs,
        seed=1024,
        init_std=1e-4,
        device="cpu",
        output_fn=torch.sigmoid,
    ):
        super(LR, self).__init__()
        torch.manual_seed(seed)
        self.sparse_defs = sparse_defs
        self.dense_defs = dense_defs
        self.output_fn = output_fn
        self.device = device
        # sparse definition
        self.embedding_layer = nn.ModuleDict(
            {
                feat.name: nn.Embedding(
                    num_embeddings=feat.vocabulary_size,
                    embedding_dim=feat.embedding_dim,
                    sparse=False,
                )
                for feat in self.sparse_defs
            }
        )
        for tensor in self.embedding_layer.values():
            nn.init.normal_(tensor.weight, mean=0, std=init_std)
        self.embedding_layer = self.embedding_layer.to(self.device)
        # dense definition
        self.weight = nn.Parameter(
            torch.Tensor(sum(fc.dimension for fc in self.dense_defs), 1).to(self.device)
        )
        torch.nn.init.normal_(self.weight, mean=0, std=init_std)
        # module locale
        self.to(self.device)

    def forward(self, X):
        # sparse part
        sparse_embeds = [
            self.embedding_layer[feat.name](
                X[:, feat.column_idx : feat.column_idx + 1].long()
            )
            for feat in self.sparse_defs
        ]
        sparse_embeds_cat = torch.cat(sparse_embeds, dim=-1)
        sparse_logit = torch.sum(sparse_embeds_cat, dim=-1, keepdim=False)
        # dense part
        dense_values = [
            X[:, feat.column_idx : feat.column_idx + feat.dimension]
            for feat in self.dense_defs
        ]
        dense_logit = torch.cat(dense_values, dim=-1).matmul(self.weight)
        # output
        output = self.output_fn(dense_logit + sparse_logit)
        return output


def train_lr_dist(num_workers=2, use_gpu=False, rand_seed=2021):
    dataloader = DataLoader("examples/dataset/ctr/criteo_100k.txt")
    dataloader = (
        dataloader.set_label_column(label_name="label")
        .set_dense_features(feat_names=[f"I{i}" for i in range(1, 14)])
        .set_sparse_features(
            feat_names=[f"C{i}" for i in range(1, 27)], embedding_dim=1
        )
        # this order should follow the data file
        .set_features_order(order=("dense", "sparse"))
    )
    datasets = dataloader.split()
    dense_defs = dataloader.dense_defs
    sparse_defs = dataloader.sparse_defs
    torch_dataset_options = dataloader.gen_torch_dataset_options()

    # dense_defs is like,
    # [{'name': 'I1', 'dimension': 1.0, 'dtype': 'float32', 'column_idx': 1.0,
    #   'feat_type': 'DenseFeat'},
    #  {'name': 'I2', 'dimension': 1.0, 'dtype': 'float32', 'column_idx': 2.0,
    #   'feat_type': 'DenseFeat'}, ...]

    # sparse_defs is like,
    # [{'name': 'C1', 'vocabulary_size': 557.0, 'embedding_dim': 1.0,
    #   'dtype': 'int32', 'group_name': 'default_group', 'column_idx': 14.0,
    #   'feat_type': 'SparseFeat'},
    #  {'name': 'C2', 'vocabulary_size': 507.0, 'embedding_dim': 1.0,
    #   'dtype': 'int32', 'group_name': 'default_group', 'column_idx': 15.0,
    #   'feat_type': 'SparseFeat'}, ...]

    # launch the trainer
    trainer = NeuralNetTrainer(
        # module and dataset configs
        module=LR,
        module_params={
            "dense_defs": dense_defs,
            "sparse_defs": sparse_defs,
            "seed": rand_seed,
        },
        dataset=datasets,
        dataset_options=torch_dataset_options,
        # trainer configs
        epochs=20,
        batch_size=512,
        loss_fn=nn.BCELoss(),
        optimizer=FTRL,
        metric_fns=[auroc],
        num_workers=num_workers,
        use_gpu=use_gpu,
        callbacks=["json", "tbx"],
    )
    results = trainer.run()
    pprint_results(results)


if __name__ == "__main__":
    environ_validate(n_cpus=1 + 2)
    train_lr_dist(num_workers=2)

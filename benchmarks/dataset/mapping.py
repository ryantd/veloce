import os

wd = os.path.dirname(os.path.realpath(__file__))

criteo_mini_feat = dict(
    sparse=[f"C{i}" for i in range(1, 27)],
    dense=[f"I{i}" for i in range(1, 14)],
    label="label",
    path=f"{wd}/ctr/criteo_mini.txt",
)

BUILTIN_DATASET_FEATS_MAPPING = {"criteo_mini": criteo_mini_feat}

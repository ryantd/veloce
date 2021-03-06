> ‼️ Please note that this library may be recently changed its name and repository address ‼️

<br />
<div align="center">
  <a href="https://github.com/ryantd/veloce">
    <img src="docs/images/logo.png" alt="Veloce Logo">
  </a>
</div>
<br />

> The project is currently under heavy development, and focusing on **PyTorch** and the **recommendation** scenario.

## About
**Veloce** is an `instant distributed computing` library based on the **Ray Train** and **Ray Data**, which is scalable, efficient, and easy-to-use. It accelerates the development of any ML/DL training workload, on any cloud or local, at any parallelism size. Previously named Enscale.

### Goals
- Launch any interactive ML/DL workloads **instantly** on your laptop or to any cloud
- **Scale** your own single-machine neural network modules to a native **distributed** manner
- Apply **heterogeneous** architecture
- **Data scientist-friendly** API
- **Sparse** and **dense feature** definitions

### Non-Goals
- Not a neural network library, there are only some benchmark modules provided.

## Getting Started
### Prerequisites
- Python version >= `3.7.1`
- Packages
  - requests >= `2.26.0`
  - ray >= `1.9.2` and <= `1.10`
  - torch >= `1.9.1`
  - pandas >= `1.3.5`
  - pyarrow >= `6.0.1`

### Installation

#### Using Pre-compiled Wheels
```shell
# CPU version
pip install veloce
```

#### From Source
```shell
git clone https://github.com/ryantd/veloce
cd veloce
pip install -e .
```

### Runtime environment

The library can launch **locally** or on any **cloud provider** with Ray set up.

- If you want to launch on the cloud, go through this [doc](https://docs.ray.io/en/latest/cluster/cloud.html#launching-cloud-clusters) to set up your Ray Cluster. And then you can use `environ_validate(n_cpus=N, cluster_endpoint="ray://<head_node_host>:<port>")` to connect your cluster.
- Or just use `environ_validate(n_cpus=N)` to have a local experience.

You can add more native `ray.init` arguments, just put them into `environ_validate` call. Like `environ_validate(n_cpus=N, ignore_reinit_error=True)` to make Ray suppresses errors from calling `ray.init()` a second time.

### Lightning example

> See more hands-on and advanced examples [here](examples/readme.md), like heterogeneous support and sparsity definition.

The following example requires `sklearn` to be installed. And `tqdm` is optional, which enables progress reporting.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1mtDacq6Yty4k5tz_6iVA-lk83YnD46S3?usp=sharing)

```python
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from veloce.util import pprint_results, load_benchmark_dataset
from veloce.model.ctr import DeepFM
from veloce import NeuralNetTrainer, environ_validate

N_WORKERS = 2
N_DATA_PROCESSOR = 1

# ray environment setup
environ_validate(n_cpus=N_DATA_PROCESSOR + N_WORKERS)
# load dataset and sparsity definition pre-defined
datasets, feature_defs, dataset_options = load_benchmark_dataset(
    # set your own dataset by `data_path="criteo_mini.txt"`
    separate_valid_dataset=False
)
# trainer setup
trainer = NeuralNetTrainer(
    # module and dataset configs
    module=DeepFM, # your own nn.Module or built in modules
    module_params={
        "dense_feature_defs": feature_defs["dense"],
        "sparse_feature_defs": feature_defs["sparse"],
    },
    dataset=datasets,
    dataset_options=dataset_options,
    # trainer configs
    epochs=5,
    batch_size=512,
    loss_fn=nn.BCELoss(),
    optimizer=torch.optim.Adam,
    metric_fns=[roc_auc_score],
    # logger callbacks
    callbacks=["json"],
    # computation abstract on distributed
    num_workers=N_WORKERS,
)
# run and print results
results = trainer.run()
pprint_results(results)
```

## Architecture

![arch](docs/images/arch.png)

## Roadmap

- Heterogeneous Strategy on Distributed Training
  - [x] Sync Parameter Server
  - [ ] Aync Parameter Server
  - [ ] Hybird Phase 1: use sync or async for the dense or sparse component as you like, under homogeneous architecture
  - [ ] Hybird Phase 2: you can choose async PS for the sparse component, and sync Ring Allreduce (like PyTorch's DDP) for the dense component
- Framework Support
  - [x] PyTorch: no specific plan to support other frameworks
- Advanced Parallel Mechanism
  - [ ] Heavy integrated [torchrec](https://github.com/pytorch/torchrec)
- Accelerator Support
  - [ ] GPU: complete inspection required

## Reference
- [Ray and Ray Train](https://github.com/ray-project/ray): Ray Train is a lightweight library for distributed deep learning, allowing you to scale up and speed up training for your deep learning models. Docs [here](https://docs.ray.io/en/master/train/train.html).
- [DeepCTR-Torch](https://github.com/shenweichen/DeepCTR-Torch): Easy-to-use, modular and extendible package of deep-learning based CTR models.

## License
Veloce is MIT licensed, as found in the [LICENSE](LICENSE) file.

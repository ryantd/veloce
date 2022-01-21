<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="">
    <img src="docs/images/logo.png" alt="Logo">
  </a>

  <h3 align="center">phetware</h3>

  <p align="center">
    An instant distributed computing ML/DL toolbox based on Ray Stack!
    <br />
    <a href=""><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="">View Demo</a>
    ·
    <a href="">Report Bug</a>
    ·
    <a href="">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-phetware">About phetware</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About `phetware`
__phetware__ is an `instant distributed computing` toolbox based on Ray Stack and ML/DL frameworks, which is scalable, efficient, and easy-to-use. It accelerates the development of any ML/DL training workload, on any cloud or local, at any number of workloads.

- One-line change to __instantly run__ workloads on any cloud provider or local laptop

- State-of-the-art like __out of box__ scaffoldings: train_fn and native NN architecture

- Flexible __modular__ to fit different use cases: core components, layers which can be used to build your own custom model/train_fn easily.

<!-- GETTING STARTED -->
## Getting Started

### Installation

- install through pip
```
pip install -U phetware
```

### Feature

#### Optimizer Stack
```python
from phetware.optimizer import OptimizerStack

trainer = NeuralNetTrainer(
  ...,
  optimizer=OptimizerStack(
    # support multiple optimizers
    dict(cls=torch.optim.Adagrad, model_key="deep_model"),
    dict(
        cls=FTRL,
        args=dict(alpha=1.0, beta=1.0, l1=1.0, l2=1.0),
        model_key="wide_model",
    ),
  ),
)
```
#### Loss Function Stack
```python
from phetware.loss_fn import LossFnStack

trainer = NeuralNetTrainer(
  ...,
  loss_fn=LossFnStack(
      # support multiple loss functions with fixed weight
      dict(fn=nn.BCELoss(), weight=0.2),
      dict(fn=nn.HingeEmbeddingLoss(), weight=0.8),
  ),
)
```

#### Early Stopping
```python
trainer = NeuralNetTrainer(
  ...,
  use_early_stopping=True,
  early_stopping_args={"patience": 2},
)
```

#### Pre-training Model
```python
trainer = NeuralNetTrainer(
  ...,
  module=FNN,
  module_params={"pre_trained_mode": True}
)

trainer.run(multi_runs=[
  {"epochs": 10},
  {"epochs": 10, "module_params": {"pre_trained_mode": False}},
])
```

<!-- USAGE EXAMPLES -->
## Usage

### `nn.Module` manner
```shell
# try `Wide and Deep`
python -m benchmarks.wdl
# try DeepFM
python -m benchmarks.deepfm
# try PNN
python -m benchmarks.pnn
# try FNN
python -m benchmarks.fnn
```

### `train_fn` manner
```shell
# try `Wide and Deep`
python -m benchmarks.wdl_raytrain
```

<!-- ROADMAP -->
## Roadmap

### CTR domain

- [x] Wide and Deep
- [x] DeepFM
- [x] FNN
- [x] PNN
- [ ] DIN
- [ ] DIEN
<!-- - [ ] AFM
- [ ] NFM
- [ ] PLM -->


<!-- CONTRIBUTING -->
## Contributing

<!-- LICENSE -->
## License

<!-- CONTACT -->
## Contact

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

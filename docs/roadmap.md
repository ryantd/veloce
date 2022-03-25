# Roadmap

## Key Features
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

## General ideas
- The project focuses on a high-level API wrapper and application based on Ray Train (will adapt the new ray.ml Trainer). The user only needs to feed the native nn.Module (without any training_loop things) to a Veloce Trainer, and he or she can set loss functions, optimizers, metric functions, etc. in Veloce Trainer as an argument. In this way, users can start their out-of-the-box training directly without any a priori knowledge more or less.
- In addition to defining loss functions, optimizers, and metric functions in Veloce Trainer, what I would like to provide is some advanced parallel strategies and methods. For example, as I mentioned before, a DDP synchronous strategy for dense parameters and a PS asynchronous strategy for sparse parameters. If possible, more complex allreduce implement variants will be integrated, like group-based ring-allreduce, or 2d ring-allreduce.
- The project will eventually not be fixed to the recommendation domain, while we are only focusing on this one scenario now because we believe that the recommendation architecture at this stage is the most complex and best validates the above strategy. The next scenario may be the GNN and DGL framework.
- We are well aware that an out-of-the-box approach may not bring model performance to SOTA, but our goal is to give developers a very intuitive solution, or perhaps we can call it a low-code solution.

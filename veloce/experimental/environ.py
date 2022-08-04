import ray


def init(n_cpus=2, n_gpus=0, **kwargs):
    ray.init(num_cpus=n_cpus, num_gpus=n_gpus, **kwargs)


def shutdown():
    ray.shutdown()

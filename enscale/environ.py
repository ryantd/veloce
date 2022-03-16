import ray

INIT_FLAG = "ENSCALE_RAY_ENVIRON_INITIALIZED"


def environ_validate(n_cpus=2, n_gpus=0, cluster_endpoint=None, **kwargs):
    ray.init(num_cpus=n_cpus, num_gpus=n_gpus, address=cluster_endpoint, **kwargs)

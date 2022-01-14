import os

import ray

INIT_FLAG = "PHETWARE_RAY_ENVIRON_INITIALIZED"


def environ_validate(num_cpus):
    if not os.environ.get(INIT_FLAG):
        ray.init(num_cpus=num_cpus)

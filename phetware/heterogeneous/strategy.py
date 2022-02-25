class Arch(object):
    PS = "parameter-server"
    RingAllreduce = "ring-allreduce"


class UpdateStrategy(object):
    Sync = "sync"
    Async = "async"


class HeterogeneousStrategy(object):
    def __init__(
        self,
        dense_arch=Arch.RingAllreduce,
        dense_update_strategy=UpdateStrategy.Sync,
        sparse_arch=Arch.RingAllreduce,
        sparse_update_strategy=UpdateStrategy.Sync,
    ):
        self.dense_arch = dense_arch
        self.dense_update_strategy = dense_update_strategy
        self.sparse_arch = sparse_arch
        self.sparse_update_strategy = sparse_update_strategy

    def use_raytrainer(self):
        if (
            self.dense_arch == Arch.RingAllreduce
            and self.dense_update_strategy == UpdateStrategy.Sync
            and self.sparse_arch == Arch.RingAllreduce
            and self.sparse_update_strategy == UpdateStrategy.Sync
        ):
            return True
        return False

    def use_sync_ps(self):
        if (
            self.dense_arch == Arch.PS
            and self.dense_update_strategy == UpdateStrategy.Sync
            and self.sparse_arch == Arch.PS
            and self.sparse_update_strategy == UpdateStrategy.Sync
        ):
            return True
        return False


class PSStrategy(HeterogeneousStrategy):
    def __init__(self, update_strategy=UpdateStrategy.Sync):
        super(PSStrategy, self).__init__(
            dense_arch=Arch.PS,
            dense_update_strategy=update_strategy,
            sparse_arch=Arch.PS,
            sparse_update_strategy=update_strategy,
        )

import inspect

import ray

TIME_DIFF = "epoch_seconds"
EARLY_STOPPED = "is_early_stopped"
CHECKPOINT_BASED = "is_checkpoint_based"


class StyleCoder(object):
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    BOLDCYAN = "\033[1;96m"
    GREEN = "\033[92m"
    BOLDGREEN = "\033[1;92m"
    WARNING = "\033[93m"
    RED = "\033[91m"
    BOLDRED = "\033[1;91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

    def __init__(self, use_style):
        self.use_style = use_style

    def __getattr__(self, code):
        if self.use_style:
            try:
                rc = getattr(self, code)
            except:
                raise AttributeError("Arg code is not valid")
        else:
            rc = ""
        return rc


def get_package_name(func):
    return get_module_name(func).split(".")[0]


def get_module_name(func):
    return inspect.getmodule(func).__name__


def get_type(func):
    return type(func).__name__


def get_func_name(func):
    return func.__name__


def inspect_func_args(config, in_func):
    in_func_args = dict()
    requires_arg = list(inspect.signature(in_func.__init__).parameters.keys())
    requires_arg.pop(0)
    for k, v in config.items():
        if k in requires_arg:
            in_func_args[k] = v
    return in_func_args


def merge_results(
    validation_result,
    train_result=None,
    time_diff=None,
    is_early_stopped=False,
    is_checkpoint_based=False,
):
    result = dict(
        is_early_stopped=is_early_stopped, is_checkpoint_based=is_checkpoint_based
    )
    if validation_result is None:
        raise ValueError("Arg validation_result should be given")
    if train_result is None:
        train_result = dict()
    for k, v in train_result.items():
        result[f"train/{k}"] = v
    for k, v in validation_result.items():
        result[f"valid/{k}"] = v
    if time_diff is not None:
        result[TIME_DIFF] = time_diff
    return result


def pprint_results(
    run_results,
    use_style=True,
    show_avg_results=True,
    show_epoch_results=True,
    show_post_analysis=True,
    print_interval=1,
):
    s = StyleCoder(use_style)
    is_es = False
    is_ckpt = False
    for run_idx, worker_results in enumerate(run_results):
        print(f"\n{s.BOLD}Run {run_idx}: {s.ENDC}")
        total = 0
        n_workers = len(worker_results or [])
        avg_metrics = {}
        for worker_idx, results in enumerate(worker_results or []):
            if not len(results):
                continue
            time_diff = results[0].pop(TIME_DIFF)
            is_es = results[0].pop(EARLY_STOPPED) or is_es
            is_ckpt = results[0].pop(CHECKPOINT_BASED) or is_ckpt
            acc_metrics = {k: v for k, v in results[0].items()}
            if len(avg_metrics.keys()) == 0:
                avg_metrics = {k: v for k, v in results[-1].items()}
            else:
                for k, v in results[-1].items():
                    avg_metrics[k] += v
            total = len(results)
            if show_epoch_results:
                print(
                    f"{s.BOLD}========================="
                    f"\nWorker {worker_idx} training results"
                    f"\n========================={s.ENDC}"
                )
            for idx in range(print_interval - 1, total, print_interval):
                val = results[idx]
                if TIME_DIFF in val:
                    time_diff = val.pop(TIME_DIFF)
                if EARLY_STOPPED in val:
                    is_es = val.pop(EARLY_STOPPED) or is_es
                if CHECKPOINT_BASED in val:
                    is_ckpt = val.pop(CHECKPOINT_BASED) or is_ckpt
                metrics_join = []
                for k, v in val.items():
                    if idx == total - 1:
                        acc_metrics[k] -= v
                    metrics_join.append(f"{k}: {'%.5f' % v}")
                metrics_join = "\t".join(metrics_join)
                print(
                    f"[epoch {idx + 1}/{total}: {'%.3f' % time_diff}s]\t{metrics_join}"
                )
            acc_metrics_join = "\t".join(
                [
                    f"{k}: {s.BOLDGREEN}↓ {s.ENDC}{'%.3f' % (v*100)}%"
                    if v > 0
                    else f"{k}: {s.BOLDRED}↑ {s.ENDC}{'%.3f' % -(v*100)}%"
                    for k, v in acc_metrics.items()
                ]
            )
            es_indicator = " ES" if is_es else ""
            ckpt_indicator = " CKPT" if is_ckpt else ""
            if show_post_analysis:
                print(
                    f"{s.BOLD}======================"
                    f"\nWorker {worker_idx} post-analysis"
                    f"\n======================{s.ENDC}"
                    f"\n[epoch 1 {s.BOLD}→{s.ENDC} {total}{s.BOLDCYAN}"
                    f"{ckpt_indicator}{es_indicator}{s.ENDC}]\t{acc_metrics_join}\n"
                )
        if show_avg_results:
            metrics_join = []
            for k, v in avg_metrics.items():
                if k in [EARLY_STOPPED, CHECKPOINT_BASED, TIME_DIFF]:
                    continue
                metrics_join.append(f"{k} avg: {'%.5f' % (v / n_workers)}")
            metrics_join = "\t".join(metrics_join)
            print(
                f"{s.BOLD}========================="
                f"\nRun average final results"
                f"\n========================={s.ENDC}"
                f"\n{metrics_join}\n"
            )


def ticks(iter):
    ops = list(iter)
    for resp in ops:
        ray.get(resp)


def calculate_batches(subset_cnt, size, n):
    batches = subset_cnt / size / n
    if batches.is_integer():
        return batches
    else:
        return int(batches) + 1


def get_batches(config, n_workers):
    return calculate_batches(
        config["dataset_options"]["train_set_count"], config["batch_size"], n_workers
    ), calculate_batches(
        config["dataset_options"]["count"]
        - config["dataset_options"]["train_set_count"],
        config["batch_size"],
        n_workers,
    )

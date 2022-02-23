import inspect

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
    if train_result is None:
        train_result = dict()
    for k, v in train_result.items():
        result[f"train/{k}"] = v
    for k, v in validation_result.items():
        result[f"valid/{k}"] = v
    if time_diff is not None:
        result[TIME_DIFF] = time_diff
    return result


def pprint_results(run_results, use_style=True, print_interval=1):
    s = StyleCoder(use_style)
    is_es = False
    is_ckpt = False
    avg_metrics = {}
    for run_idx, worker_results in enumerate(run_results):
        print(f"\n{s.BOLD}Run {run_idx}: {s.ENDC}")
        total = 0
        n_workers = len(worker_results)
        for worker_idx, results in enumerate(worker_results):
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
            print(
                f"{s.BOLD}======================"
                f"\nWorker {worker_idx} post-analysis"
                f"\n======================{s.ENDC}"
                f"\n[epoch 1 {s.BOLD}→{s.ENDC} {total}{s.BOLDCYAN}"
                f"{ckpt_indicator}{es_indicator}{s.ENDC}]\t{acc_metrics_join}\n"
            )

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

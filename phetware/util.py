import inspect

TIME_DIFF = "epoch_seconds"


class StyleCoder(object):
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
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


def merge_results(validation_result, test_result, time_diff=None):
    result = dict()
    for k, v in validation_result.items():
        result[f"valid/{k}"] = v
    for k, v in test_result.items():
        result[f"test/{k}"] = v
    if time_diff is not None:
        result[TIME_DIFF] = time_diff
    return result


def pprint_results(run_results, use_style=True, print_interval=1):
    s = StyleCoder(use_style)
    for run_idx, worker_results in enumerate(run_results):
        print(f"\n{s.BOLD}Run {run_idx}: {s.ENDC}")
        for worker_idx, results in enumerate(worker_results):
            if not len(results):
                continue
            time_diff = results[0].pop(TIME_DIFF)
            acc_metrics = {k: v for k, v in results[0].items()}
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
                metrics_join = []
                for k, v in val.items():
                    if idx == total - 1:
                        acc_metrics[k] -= v
                    metrics_join.append(f"{k}: {'%.5f' % v}")
                metrics_join = "\t".join(metrics_join)
                print(
                    f"[epoch {idx + 1}/{total}: {'%.3f' % time_diff}s]\t{metrics_join}"
                )
            acc_metrics_join = "\t".join([
                f"{k}: {s.BOLDGREEN}↓ {s.ENDC}{'%.3f' % (v*100)}%"
                if v > 0
                else f"{k}: {s.BOLDRED}↑ {s.ENDC}{'%.3f' % -(v*100)}%"
                for k, v in acc_metrics.items()
            ])
            print(
                f"{s.BOLD}================="
                f"\nWorker {worker_idx} analysis"
                f"\n================={s.ENDC}"
                f"\n[epoch 1→{total}]\t{acc_metrics_join}\n"
            )

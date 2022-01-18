import inspect

TIME_DIFF = "epoch_seconds"


class StyleCoder(object):
    def __init__(self, use_style):
        self.use_style = use_style

    def __call__(self, code):
        return f"\033[{code}m" if self.use_style else ""


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
        print(f"\n{s('1')}Run {run_idx}: {s('0')}")
        for worker_idx, results in enumerate(worker_results):
            if not len(results):
                continue
            time_diff = results[0].pop(TIME_DIFF)
            acc_metrics = {k: v for k, v in results[0].items()}
            total = len(results)
            print(
                f"{s('1')}========================="
                f"\nWorker {worker_idx} training results"
                f"\n========================={s('0')}"
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
            print(
                f"{s('1')}========================="
                f"\nWorker {worker_idx} analysis"
                f"\n========================={s('0')}"
            )
            print(
                "\t".join(
                    [
                        f"{s('1;32')}{k}: {'%.3f' % -(v*100)}%{s('0')}"
                        if v > 0
                        else f"{s('1;31')}{k}: +{'%.3f' % -(v*100)}%{s('0')}"
                        for k, v in acc_metrics.items()
                    ]
                )
            )
    print("\n")

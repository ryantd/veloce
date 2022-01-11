import inspect

from ray.train.trainer import S


def get_package_name(func):
    return get_module_name(func).split(".")[0]


def get_module_name(func):
    return inspect.getmodule(func).__name__


def merge_results(validation_result, test_result):
    result = dict()
    for k, v in validation_result.items():
        result[f"validation/{k}"] = v
    for k, v in test_result.items():
        result[f"test/{k}"] = v
    return result


def pprint_results(all_results, use_style=True, print_interval=1):
    class StyleCoder(object):
        def __init__(self, use_style):
            self.use_style = use_style
        def __call__(self, code):
            return f"\033[{code}m" if self.use_style else ""
    s = StyleCoder(use_style)
    for worker_idx, results in enumerate(all_results):
        acc_metrics = {k: v for k, v in results[0].items()}
        total = len(results)
        print(f"\n{s('1')}=========================\nWorker {worker_idx} training results\n========================={s('0')}")
        for idx in range(print_interval-1, total, print_interval):
            val = results[idx]
            metrics_join = []
            for k, v in val.items():
                if idx == total - 1: acc_metrics[k] -= v
                metrics_join.append(f"{k}: {'%.5f' % v}")
            metrics_join = "\t".join(metrics_join)
            print(f"[epoch {idx + 1}/{total}]\t{metrics_join}")
        print(f"{s('1')}=========================\nWorker {worker_idx} analysis\n========================={s('0')}")
        print("\t".join([f"{s('1;32')}{k}: {'%.3f' % -(v*100)}%{s('0')}" if v > 0 else f"{s('1;31')}{k}: +{'%.3f' % -(v*100)}%{s('0')}" for k, v in acc_metrics.items()]))
    print("\n")

import inspect


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

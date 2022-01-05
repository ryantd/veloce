import inspect


def get_package_name(func):
    return get_module_name(func).split(".")[0]


def get_module_name(func):
    return inspect.getmodule(func).__name__

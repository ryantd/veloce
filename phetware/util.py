import inspect


def get_package_name(func):
    module_name = inspect.getmodule(func).__name__
    return module_name.split(".")[0]
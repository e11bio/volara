from importlib import import_module

__all__ = ["datasets", "dbs", "__version__", "__version_info__"]


def __getattr__(name: str):
    if name in {"datasets", "dbs"}:
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__version__ = "1.0.5"
__version_info__ = tuple(int(i) for i in __version__.split("."))

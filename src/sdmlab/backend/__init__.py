from .numpy_backend import NumpyBackend

_active = NumpyBackend()  # default so nobody has to call set_backend() to get started

def get_backend():
    return _active

def set_backend(name: str) -> None:
    global _active
    if name == "numpy":
        _active = NumpyBackend()
    elif name == "cupy":
        try:
            from .cupy_backend import CupyBackend
        except ImportError as e:
            raise ImportError("cupy backend requires cupy — try `pip install sdmlab[gpu]`") from e
        _active = CupyBackend()
    else:
        raise ValueError(f"unknown backend: {name!r}")

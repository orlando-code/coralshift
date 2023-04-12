import importlib
import inspect
from pathlib import Path


def get_coralshift_dir():
    coralshift_module = importlib.import_module("coralshift")
    coralshift_dir = Path(inspect.getabsfile(coralshift_module)).parent
    return (coralshift_dir / "..").resolve()

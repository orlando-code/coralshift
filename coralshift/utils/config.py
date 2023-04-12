from pathlib import Path
import importlib
import inspect


def get_h3_dir():
    coralshift_module = importlib.import_module("coralshift")
    coralshift_dir = Path(inspect.getabsfile(coralshift_module)).parent
    return (coralshift_dir / "..").resolve()

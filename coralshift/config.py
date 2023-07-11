import importlib
import inspect
from pathlib import Path

# from setuptools import setup, find_packages


def get_coralshift_dir():
    coralshift_module = importlib.import_module("coralshift")
    coralshift_dir = Path(inspect.getabsfile(coralshift_module)).parent
    return (coralshift_dir / "..").resolve()


# setup(name="coralshift", version="0.1.0", packages=find_packages())

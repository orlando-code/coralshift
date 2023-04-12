import os
import importlib
import inspect
# N.B. not too sure about this functionality: just reproduced from h3


def get_coralshift_dir():
    coralshift_module = importlib.import_module("coralshift")
    coralshift_dir = os.path.dirname(inspect.getabsfile(coralshift_module))
    return os.path.abspath(os.path.join(coralshift_dir, ".."))

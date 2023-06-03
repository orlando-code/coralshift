import os

from pathlib import Path
from coralshift.utils import directories


class MyDatasets:
    """Handle the variety of datasets required to test and train model"""

    # TODO: add in declaration of filepath root
    def __init__(self):
        self.datasets = {}
        self.files_location = Path()
        # fetching external functions

    def set_location(self, location="remote"):
        if location == "remote":
            # change directory to home. TODO: make less hacky
            os.chdir("/home/jovyan")
            self.files_location = Path("lustre_scratch/datasets/")
        elif location == "local":
            self.files_location = directories.get_volume_dir()
        else:
            raise ValueError

    def get_location(self):
        return self.files_location

    def add_dataset(self, name, data):
        self.datasets[name] = data

    def add_datasets(self, names, data):
        for i, name in enumerate(names):
            self.datasets[name] = data[i]

    def get_dataset(self, name):
        return self.datasets.get(name, None)

    def remove_dataset(self, name):
        if name in self.datasets:
            del self.datasets[name]

    def list_datasets(self):
        return list(self.datasets.keys())

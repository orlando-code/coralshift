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

    def set_location(self, location: str = "remote", volume_name: str = "MRes Drive"):
        """Required to go between accessing data on remote (MAGEO) and local (usually
        external harddrive)
        """
        if location == "remote":
            # change directory to home. TODO: make less hacky
            os.chdir("/home/jovyan")
            self.files_location = Path("lustre_scratch/datasets/")
        elif location == "local":
            self.files_location = directories.get_volume_dir(volume_name) / "datasets/"
        else:
            raise ValueError(f"Unrecognised location: {location}")

    def get_location(self):
        return self.files_location

    def add_dataset(self, name: str, data):
        self.datasets[name] = data

    def add_datasets(self, names: list[str], data: list):
        for i, name in enumerate(names):
            self.datasets[name] = data[i]

    def get_dataset(self, name: str):
        return self.datasets.get(name, None)

    def remove_dataset(self, name: str):
        if name in self.datasets:
            del self.datasets[name]

    def list_datasets(self):
        return list(self.datasets.keys())

import importlib
import inspect
from pathlib import Path
import os
import pandas as pd


def get_coralshift_dir():
    coralshift_module = importlib.import_module("coralshift")
    coralshift_dir = Path(inspect.getabsfile(coralshift_module)).parent
    return (coralshift_dir / "..").resolve()


# setup(name="coralshift", version="0.1.0", packages=find_packages())


"""
Defines globals used throughout the codebase.
"""

###############################################################################
### Folder structure naming system
###############################################################################

data_folder = "data"

cmip6_data_folder = os.path.join(data_folder, "env_vars", "cmip6")
bathymetry_folder = os.path.join(data_folder, "bathymetry")
gt_folder = os.path.join(data_folder, "ground_truth")

# dataloader_config_folder = "dataloader_configs"

# networks_folder = "trained_networks"

# results_folder = "results"
# forecast_results_folder = os.path.join(results_folder, "forecast_results")
# permute_and_predict_results_folder = os.path.join(
#     results_folder, "permute_and_predict_results"
# )
# uncertainty_results_folder = os.path.join(results_folder, "uncertainty_results")

# figure_folder = "figures"

# video_folder = "videos"

# active_grid_cell_file_format = "active_grid_cell_mask_{}.npy"
# land_mask_filename = "land_mask.npy"
# region_mask_filename = "region_mask.npy"

###############################################################################
### Missing months
###############################################################################

###############################################################################
### Weights and biases config (https://docs.wandb.ai/guides/track/advanced/environment-variables)
###############################################################################

# TODO: set up wandb config
# Get API key from https://wandb.ai/authorize
WANDB_API_KEY = "YOUR-KEY-HERE"
# Absolute path to store wandb generated files (folder must exist)
#   Note: user must have write access
WANDB_DIR = "/path/to/wandb/dir"
# Absolute path to wandb config dir (
WANDB_CONFIG_DIR = "/path/to/wandb/config/dir"
WANDB_CACHE_DIR = "/path/to/wandb/cache/dir"

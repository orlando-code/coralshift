# import importlib
# import inspect
from pathlib import Path
import subprocess


# def get_coralshift_dir():
#     coralshift_module = importlib.import_module("coralshift")
#     coralshift_dir = Path(inspect.getabsfile(coralshift_module)).parent
#     return (coralshift_dir / "..").resolve()
# setup(name="coralshift", version="0.1.0", packages=find_packages())


# def get_coralshift_module_dir():
#     return Path(__file__).resolve().parent.parent


# def get_repo_dir():
#     return get_coralshift_module_dir().parent


def get_repo_root():
    # Run 'git rev-parse --show-toplevel' command to get the root directory of the Git repository
    git_root = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True
    )
    if git_root.returncode == 0:
        return Path(git_root.stdout.strip())
    else:
        raise RuntimeError("Unable to determine Git repository root directory.")


"""
Defines globals used throughout the codebase.
"""

###############################################################################
# Folder structure naming system
###############################################################################

repo_dir = get_repo_root()
data_dir = repo_dir / "data"
logs_dir = repo_dir / "logs"
runs_dir = repo_dir / "runs" / "first_run"
runs_csv = runs_dir / "runs.csv"

cmip6_data_dir = data_dir / "env_vars" / "cmip6"
static_cmip6_data_dir = cmip6_data_dir / "EC-Earth3P-HR/r1i1p2f1_latlon"
bathymetry_dir = data_dir / "bathymetry"
gt_data_dir = data_dir / "ground_truth"
# subdirs
gdcr_dir = gt_data_dir / "unep_wcmc"


# dataloader_config_dir = "dataloader_configs"

# networks_dir = "trained_networks"

# results_dir = "results"
# forecast_results_dir = results_dir, "forecast_results")
# permute_and_predict_results_dir =
#     results_dir, "permute_and_predict_results"
# )
# uncertainty_results_dir = results_dir, "uncertainty_results")

# figure_dir = "figures"

# video_dir = "videos"

# active_grid_cell_file_format = "active_grid_cell_mask_{}.npy"
# land_mask_filename = "land_mask.npy"
# region_mask_filename = "region_mask.npy"

###############################################################################
# Missing months
###############################################################################

###############################################################################
# Weights and biases config (https://docs.wandb.ai/guides/track/advanced/environment-variables)
###############################################################################

# TODO: set up wandb config
# Get API key from https://wandb.ai/authorize
WANDB_API_KEY = "b31d9bc1dad7625997b86c2a5db38f8369babbe7"
# Absolute path to store wandb generated files (dir must exist)
#   Note: user must have write access
WANDB_DIR = "/maps/rt582/coralshift/wandb"
# Absolute path to wandb config dir (
WANDB_CONFIG_DIR = "/path/to/wandb/config/dir"
WANDB_CACHE_DIR = "/path/to/wandb/cache/dir"

# import importlib
# import inspect
from pathlib import Path
import subprocess
from dataclasses import dataclass


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

# REPO DIRECTORIES
repo_dir = get_repo_root()
module_dir = repo_dir / "coralshift"
data_dir = repo_dir / "data"
logs_dir = repo_dir / "logs"
runs_dir = repo_dir / "runs"

# DATA DIRECTORIES
bathymetry_dir = data_dir / "bathymetry"
env_data_dir = data_dir / "env_vars"
gt_data_dir = data_dir / "ground_truth"
ml_ready_dir = data_dir / "ml_ready"
# DATA SUBDIRECTORIES
wri_dir = gt_data_dir / "WRI_REEF_EXTENT"
gdcr_dir = gt_data_dir / "UNEP_GDCR"
cmip6_data_dir = env_data_dir / "cmip6"
static_cmip6_data_dir = cmip6_data_dir / "EC-Earth3P-HR/r1i1p2f1_latlon"

# RUNS DIRECTORIES
runs_csv = runs_dir / "runs.csv"    # run metadata file
models_dir = runs_dir / "models"    # trained model directory
config_dir = runs_dir / "config_files"
run_logs_dir = logs_dir / "runs"    # logging directory


###############################################################################
# Config classes
###############################################################################

@dataclass
class ProcessingConfig:
    do_crop: bool

    def __init__(self, conf: dict):
        self.do_crop = conf['do_crop']

@dataclass
class HyperparameterSearchConfig:
    cv_folds: int
    n_samples: int
    n_iter: int
    search_type: str
    do_search: bool
    n_trials: int
    search_types: list[str]
    n_jobs: int

    def __init__(self, conf: dict):
        self.cv_folds = conf['cv_folds']
        self.n_samples = conf['n_samples']
        self.n_iter = conf['n_iter']
        self.search_type = conf['search_type']
        self.do_search = conf['do_search']
        self.n_trials = conf['n_trials']
        self.search_types = conf['search_types']
        self.n_jobs = conf['n_jobs']

@dataclass
class Config:
    # could use a dynamic class here, but these are required parameters so will keep static for now
    # https://alexandra-zaharia.github.io/posts/python-configuration-and-dataclasses/#dynamically-creating-a-configuration-class
    data_source: str
    regressor_classification_threshold: float
    depth_mask: list[int]
    ds_type: str
    predictand: str
    datasets: list[str]
    env_vars: list[str]
    random_state: int
    resolution: float
    resolution_unit: str
    year_range_to_include: list[int]
    upsample_method: str
    downsample_method: str
    spatial_buffer: int
    save_figs: bool
    do_train: bool
    do_save_model: bool
    do_plot: bool
    processing: ProcessingConfig
    lats: list[int]
    lons: list[int]
    levs: list[int]
    split_type: str
    test_geom: list[int]
    train_test_val_frac: list[float]
    X_scaler: str
    y_scaler: str
    hyperparameter_search: HyperparameterSearchConfig
    source_id: str
    member_id: str

    def __init__(self, conf: dict):
        self.data_source = conf['data_source']
        self.regressor_classification_threshold = conf['regressor_classification_threshold']
        self.depth_mask = conf['depth_mask']
        self.ds_type = conf['ds_type']
        self.predictand = conf['predictand']
        self.datasets = conf['datasets']
        self.env_vars = conf['env_vars']
        self.random_state = conf['random_state']
        self.resolution = conf['resolution']
        self.resolution_unit = conf['resolution_unit']
        self.year_range_to_include = conf['year_range_to_include']
        self.upsample_method = conf['upsample_method']
        self.downsample_method = conf['downsample_method']
        self.spatial_buffer = conf['spatial_buffer']
        self.save_figs = conf['save_figs']
        self.do_train = conf['do_train']
        self.do_save_model = conf['do_save_model']
        self.do_plot = conf['do_plot']
        self.processing = ProcessingConfig(conf['processing'])
        self.lats = conf['lats']
        self.lons = conf['lons']
        self.levs = conf['levs']
        self.split_type = conf['split_type']
        self.test_geom = conf['test_geom']
        self.train_test_val_frac = conf['train_test_val_frac']
        self.X_scaler = conf['X_scaler']
        self.y_scaler = conf['y_scaler']
        self.hyperparameter_search = HyperparameterSearchConfig(conf['hyperparameter_search'])
        self.source_id = conf['source_id']
        self.member_id = conf['member_id']
        

# ###############################################################################
# # Weights and biases config (https://docs.wandb.ai/guides/track/advanced/environment-variables)
# ###############################################################################

# # TODO: set up wandb config
# # Get API key from https://wandb.ai/authorize
# WANDB_API_KEY = "b31d9bc1dad7625997b86c2a5db38f8369babbe7"
# # Absolute path to store wandb generated files (dir must exist)
# #   Note: user must have write access
# WANDB_DIR = "/maps/rt582/coralshift/wandb"
# # Absolute path to wandb config dir (
# WANDB_CONFIG_DIR = "/path/to/wandb/config/dir"
# WANDB_CACHE_DIR = "/path/to/wandb/cache/dir"

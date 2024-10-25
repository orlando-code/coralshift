from pathlib import Path


def get_cmipper_module_dir():
    return Path(__file__).resolve().parent


def get_repo_dir():
    return get_cmipper_module_dir().parent


# Define global filepaths used throughout
################################################################################

repo_dir = get_repo_dir()
data_dir = repo_dir / "data"
logging_dir = repo_dir / "logs"
model_info = repo_dir / "model_info.yaml"
download_config = repo_dir / "download_config.yaml"

cmip6_data_dir = data_dir / "env_vars" / "cmip6"

# TODO: automate creation of example figures and videos from downloads. But who has the time?
# figure_folder = "figures"
# video_folder = "videos"

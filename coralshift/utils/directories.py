from __future__ import annotations

from coralshift.utils import file_ops
from coralshift.config import get_coralshift_dir
from pathlib import Path

# def get_data_dir() -> Path:
#     """./data"""
#     return file_ops.guarantee_existence(get_coralshift_dir(), 'data')


def get_volume_dir(volume_name: str) -> Path:
    """~/Volumes/volume_name"""
    return Path("/Volumes", volume_name)


def get_datasets_dir() -> Path:
    """./data/datasets"""
    return file_ops.guarantee_existence(get_coralshift_dir().parent / "datasets")


def get_processed_dir() -> Path:
    """./data/datasets/processed"""
    return file_ops.guarantee_existence(get_datasets_dir() / "processed")


def get_reef_baseline_dir() -> Path:
    """./data/datasets/reef_baseline"""
    return file_ops.guarantee_existence(get_datasets_dir() / "reef_baseline")


def get_gt_files_dir() -> Path:
    """./data/datasets/reef_baseline/gt_files"""
    return file_ops.guarantee_existence(get_reef_baseline_dir() / "gt_files")


def get_cmems_dir() -> Path:
    """./data/datasets/global_ocean_reanalysis"""
    return file_ops.guarantee_existence(get_datasets_dir() / "global_ocean_reanalysis")


def get_monthly_cmems_dir() -> Path:
    """./data/datasets/global_ocean_reanalysis/monthly_means"""
    return file_ops.guarantee_existence(get_cmems_dir() / "monthly_means")


def get_daily_cmems_dir() -> Path:
    """./data/datasets/global_ocean_reanalysis/daily_means"""
    return file_ops.guarantee_existence(get_cmems_dir() / "daily_means")


def get_bathymetry_datasets_dir() -> Path:
    """./datasets/bathymetry"""
    return file_ops.guarantee_existence(get_datasets_dir() / "bathymetry")


def get_gradients_dir() -> Path:
    """./datasets/gradients"""
    return file_ops.guarantee_existence(get_datasets_dir() / "gradients")


def get_gbr_bathymetry_data_dir() -> Path:
    """./datasets/bathymetry/GBR_30m"""
    return file_ops.guarantee_existence(get_bathymetry_datasets_dir() / "GBR_30m")


def get_era5_data_dir() -> Path:
    """./datasets/era5"""
    return file_ops.guarantee_existence(get_datasets_dir() / "era5")


def get_comparison_dir() -> Path:
    """./datasets/comparison"""
    return file_ops.guarantee_existence(get_datasets_dir() / "comparison_resolutions")

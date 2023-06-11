from __future__ import annotations

from coralshift.utils.file_ops import guarantee_existence
from coralshift.config import get_coralshift_dir
from pathlib import Path

# def get_data_dir() -> Path:
#     """./data"""
#     return guarantee_existence(get_coralshift_dir(), 'data')


def get_volume_dir(volume_name: str) -> Path:
    """~/Volumes/volume_name"""
    return Path("/Volumes", volume_name)


def get_datasets_dir() -> Path:
    """./data/datasets"""
    return guarantee_existence(get_coralshift_dir().parent / "datasets")


def get_reef_baseline_dir() -> Path:
    """./data/datasets/reef_baseline"""
    return guarantee_existence(get_datasets_dir() / "reef_baseline")


def get_bathymetry_datasets_dir() -> Path:
    """./datasets/bathymetry"""
    return guarantee_existence(get_datasets_dir() / "bathymetry")


def get_gbr_bathymetry_data_dir() -> Path:
    """./datasets/bathymetry/GBR_30m"""
    return guarantee_existence(get_bathymetry_datasets_dir() / "GBR_30m")

from coralshift.file_ops import guarantee_existence
from coralshift.config import get_coralshift_dir
from pathlib import Path

# def get_data_dir() -> Path:
#     """./data"""
#     return guarantee_existence(get_coralshift_dir(), 'data')


def get_datasets_dir() -> Path:
    """./data/datasets"""
    return guarantee_existence(get_coralshift_dir(), 'datasets')


def get_reef_baseline_dir() -> Path:
    """./data/datasets/reef_baseline"""
    return guarantee_existence(get_datasets_dir(), 'reef_baseline')

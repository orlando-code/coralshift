from coralshift.dataloading import get_data, config
from coralshift import functions_creche
from coralshift.processing import spatial_data
from coralshift.utils import file_ops, utils

from sklearn.preprocessing import MinMaxScaler
import sklearn.metrics as sklmetrics
from dask import dataframe as dd

import os
from pathlib import Path
import pickle
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import xarray as xa
import rasterio
import rioxarray as rio

import importlib
import inspect


from tqdm.auto import tqdm

# file_ops.py
####################################################################################################


def generate_filepath(
    dir_path: str | Path, filename: str = None, suffix: str = None
) -> Path:
    """Generates directory path if non-existant; if filename provided, generates filepath, adding suffix if
    necessary."""
    # if generating/ensuring directory path
    if not filename:
        return config.guarantee_existence(dir_path)
    # if filename provided, seemingly with suffix included
    elif not suffix:
        return Path(dir_path) / filename
    # if filename and suffix provided
    else:
        return (Path(dir_path) / filename).with_suffix(file_ops.pad_suffix(suffix))


def merge_save_nc_files(download_dir: Path | str, filename: str):
    # read relevant .nc files and merge to return master xarray
    xa_ds = load_merge_nc_files(Path(download_dir))
    save_path = Path(Path(download_dir), filename).with_suffix(".nc")
    xa_ds.to_netcdf(save_path)
    print(f"Combined nc file written to {save_path}.")
    return xa_ds


# hopefully isn't necessary, since loading all datasets into memory is very intensive and not scalable
def naive_nc_merge(dir: Path | str):
    file_paths = file_ops.return_list_filepaths(dir, ".nc")
    # get names of files
    filenames = [file_path.stem for file_path in file_paths]
    # load in files to memory as xarray
    das = [xa.load_dataset(file_path) for file_path in file_paths]

    combined = xa.merge([da for da in das], compat="override")

    # save combined file
    save_name = filenames[0] + "&" + filenames[-1] + "_merged.nc"
    save_path = Path(dir) / save_name
    combined.to_netcdf(path=save_path)
    print(f"{save_name} saved successfully")


def load_merge_nc_files(
    nc_dir: Path | str, incl_subdirs: bool = True, concat_dim: str = "time"
):
    """Load and merge all netCDF files in a directory.

    Parameters
    ----------
        nc_dir (Path | str): directory containing the netCDF files to be merged.

    Returns
    -------
        xr.Dataset: merged xarray Dataset object containing the data from all netCDF files.
    """
    # specify whether searching subdirectories as well
    files = file_ops.return_list_filepaths(nc_dir, ".nc", incl_subdirs)
    # if only a single file present (no need to merge)
    if len(files) == 1:
        return xa.open_dataset(files[0])
    # combine nc files by time
    ds = xa.open_mfdataset(
        files,
        decode_cf=False,
        concat_dim=concat_dim,
        combine="nested",
        coords="minimal",
    )
    return xa.decode_cf(ds).sortby("time", ascending=True)


def check_file_exists(
    filepath: Path | str = None,
    dir_path: Path | str = None,
    filename: str = None,
    suffix: str = None,
) -> bool:
    """Check if a file with the specified filename and optional suffix exists in the given directory.
    # TODO: potentially more checking required
    Parameters
    ----------
    dir_path (Path | str): Path to the directory where the file should be located.
    filename (str): Name of the file to check for.
    suffix (str, optional) Optional suffix for the filename, by default None.

    Returns
    -------
    bool: True if the file exists, False otherwise.
    """
    # if filepath argument not provided, try to create from directory path and filename
    if not filepath:
        filepath = Path(dir_path) / filename
    # if suffix argument provided (likely in conjunction with a "filename", append)
    if suffix:
        filepath = Path(filepath).with_suffix(file_ops.pad_suffix(suffix))
    return filepath.is_file()


def save_nc(
    save_dir: Path | str,
    filename: str,
    xa_d: xa.DataArray | xa.Dataset,
    return_array: bool = False,
) -> xa.DataArray | xa.Dataset:
    """
    Save the given xarray DataArray or Dataset to a NetCDF file iff no file with the same
    name already exists in the directory.
    # TODO: issues when suffix provided
    Parameters
    ----------
        save_dir (Path or str): The directory path to save the NetCDF file.
        filename (str): The name of the NetCDF file.
        xa_d (xarray.DataArray or xarray.Dataset): The xarray DataArray or Dataset to be saved.

    Returns
    -------
        xarray.DataArray or xarray.Dataset: The input xarray object.
    """
    filename = file_ops.remove_suffix(utils.replace_dot_with_dash(filename))
    save_path = (Path(save_dir) / filename).with_suffix(".nc")
    if not save_path.is_file():
        if "grid_mapping" in xa_d.attrs:
            del xa_d.attrs["grid_mapping"]
        print(f"Writing {filename} to file at {save_path}")
        spatial_data.process_xa_d(xa_d).to_netcdf(save_path)
        print("Writing complete.")
    else:
        print(f"{filename} already exists in {save_dir}")

    if return_array:
        return save_path, xa.open_dataset(save_path, decode_coords="all")
    else:
        return save_path


def save_dict_xa_ds_to_nc(
    xa_d_dict: dict, save_dir: Path | str, target_resolution: float = None
) -> None:
    for filename, array in tqdm(xa_d_dict.items(), desc="Writing tifs to nc files"):
        if target_resolution:
            spatial_data.upsample_xarray_to_target(
                xa_array=array, target_resolution=target_resolution
            )
        save_nc(save_dir, filename, array)


def resample_dir_ncs(ncs_dir, target_resolution_d=1 / 27):
    nc_files = file_ops.return_list_filepaths(ncs_dir, ".nc", incl_subdirs=False)
    res_string = utils.generate_resolution_str(target_resolution_d)

    save_dir = config.guarantee_existence(ncs_dir / f"{res_string}_arrays")
    for nc in nc_files:
        nc_xa = file_ops.open_xa_file(nc).astype("float32")
        new_name = f"{str(nc.stem)}_{res_string}"
        # resample to res
        resampled = spatial_data.resample_xarray_to_target(
            xa_d=nc_xa, target_resolution_d=target_resolution_d, name=new_name
        )
        # save
        save_nc(save_dir, new_name, resampled)


def open_xa_file(xa_path: Path | str) -> xa.Dataset | xa.DataArray:
    try:
        return spatial_data.process_xa_d(
            xa.open_dataarray(xa_path, decode_coords="all")
        )
    except ValueError:
        return spatial_data.process_xa_d(xa.open_dataset(xa_path, decode_coords="all"))


def resample_list_xa_ds_to_target_resolution_and_merge(
    xa_das: list[xa.DataArray],
    target_resolution: float,
    unit: str = "m",
    lat_lims: tuple[float] = (-10, -17),
    lon_lims: tuple[float] = (142, 147),
) -> dict:
    """
    Resample a list of xarray DataArrays to the target resolution and merge them.

    Parameters
    ----------
        xa_das (list[xa.DataArray]): A list of xarray DataArrays to be resampled and merged.
        target_resolution (float): The target resolution for resampling.
        unit (str, defaults to "m"): The unit of the target resolution.
        interp_method: (str, defaults to "linear") The interpolation method for resampling.

    Returns
    -------
        A dictionary containing the resampled xarray DataArrays merged by their names.
    """
    # TODO: will probably need to save to individual files/folders and combine at test/train time
    # may need to go to target array here
    target_resolution_d = spatial_data.choose_resolution(target_resolution, unit)[1]

    dummy_xa = spatial_data.generate_dummy_xa(target_resolution_d, lat_lims, lon_lims)

    resampled_xa_das_dict = {}
    for xa_da in tqdm(xa_das, desc="Resampling xarray DataArrays"):
        xa_resampled = spatial_data.resample_xa_d_to_other(
            xa_da, dummy_xa, name=xa_da.name
        )
        # xa_resampled = spatial_data.upsample_xarray_to_target(xa_da, target_resolution_d)
        resampled_xa_das_dict[xa_da.name] = xa_resampled

    return resampled_xa_das_dict, unit


# def resample_list_xa_ds_to_target_res_and_save(
#     xa_das: list[xa.DataArray],
#     target_resolution_d: float,
#     unit: str = "m",
#     lat_lims: tuple[float] = (-10, -17),
#     lon_lims: tuple[float] = (142, 147),
# ) -> None:
#     """
#     Resamples a list of xarray DataArrays to a target resolution, and saves the resampled DataArrays to NetCDF files.

#     Parameters
#     ----------
#         xa_das (list[xa.DataArray]): A list of xarray DataArrays to be resampled.
#         target_resolution_d (float): The target resolution in degrees or meters, depending on the unit specified.
#         unit (str, optional): The unit of the target resolution. Defaults to "m".
#         lat_lims (tuple[float], optional): Latitude limits for the dummy DataArray used for resampling.
#             Defaults to (-10, -17).
#         lon_lims (tuple[float], optional): Longitude limits for the dummy DataArray used for resampling.
#             Defaults to (142, 147).

#     Returns
#     -------
#         None
#     """

# dummy_xa = spatial_data.generate_dummy_xa(target_resolution_d, lat_lims, lon_lims)

# save_dir = generate_filepath(
#     (
#         directories.get_comparison_dir()
#         / utils.replace_dot_with_dash(f"{target_resolution_d:.05f}d_arrays")
#     )
# )

# for xa_da in tqdm(
#     xa_das,
#     desc=f"Resampling xarray DataArrays to {target_resolution_d:.05f}d",
#     position=1,
#     leave=True,
# ):
#     filename = utils.replace_dot_with_dash(
#         f"{xa_da.name}_{target_resolution_d:.05f}d"
#     )
#     save_path = (save_dir / filename).with_suffix(".nc")

#     if not save_path.is_file():
#         xa_resampled = spatial_data.process_xa_d(
#             spatial_data.upsample_xa_d_to_other(
#                 spatial_data.process_xa_d(xa_da), dummy_xa, name=xa_da.name
#             )
#         )
#         # causes problems with saving
#         if "grid_mapping" in xa_resampled.attrs:
#             del xa_resampled.attrs["grid_mapping"]

#         xa_resampled.to_netcdf(save_path)
#     else:
#         print(f"{filename} already exists in {save_dir}")


# def tifs_to_resampled_ncs(
#     tifs_dir: Path | str = None,
#     target_resolution_d: float = 1 / 27,
# ):
#     if not tifs_dir:
#         tifs_dir = (directories.get_reef_baseline_dir(),)
#     tif_files = file_ops.return_list_filepaths(tifs_dir, ".tif")

#     res_string = utils.generate_resolution_str(target_resolution_d)

#     save_dir = config.guarantee_existence(tifs_dir / f"{res_string}_arrays")
#     for tif in tif_files:
#         c_xa = open_xa_file(tif)
#         new_name = f"{str(c_xa.stem)}_{res_string}"
#         # resample to res
#         resampled = spatial_data.resample_xarray_to_target(
#             xa_d=c_xa, target_resolution_d=target_resolution_d, name=new_name
#         )
#         # save
#         save_nc(save_dir, new_name, resampled)


def tif_to_xa_array(tif_path) -> xa.DataArray:
    return spatial_data.process_xa_d(rio.open_rasterio(rasterio.open(tif_path)))


# dataloading/reef_extent.py
####################################################################################################


# def process_nc_dir_coral_gt_tifs(tif_dir_name=None, target_resolution_d: float = None):
#     # if not tif_dir_name:
#     #     tif_dir = directories.get_reef_baseline_dir()
#     # else:
#     #     tif_dir = directories.get_reef_baseline_dir() / tif_dir_name

#     nc_dir = file_ops.guarantee_existence(tif_dir / "gt_nc_dir")
#     # save tifs to ncs in new dir
#     tif_paths = tifs_to_ncs(nc_dir, target_resolution_d)
#     # get list of nc paths in dir
#     xa_arrays_list = [tif_to_xa_array(tif_path) for tif_path in tif_paths]
#     # merge ncs into one mega nc file
#     if len(xa_arrays_list) > 1:
#         concatted = xa.concat(xa_arrays_list, dim=["latitude", "longitude"])
#     else:
#         concatted = xa_arrays_list[0]
#     file_ops.save_nc(
#         nc_dir, f"concatenated_{target_resolution_d:.05f}_degree", concatted
#     )


def tifs_to_ncs(nc_dir: Path | list[str], target_resolution_d: float = None) -> None:

    tif_dir = nc_dir.parent
    tif_paths = file_ops.return_list_filepaths(tif_dir, ".tif")
    xa_array_dict = {}
    for tif_path in tqdm(
        tif_paths, total=len(tif_paths), desc="Writing tifs to nc files"
    ):
        # filename = str(file_ops.get_n_last_subparts_path(tif, 1))
        filename = tif_path.stem
        tif_array = tif_to_xa_array(tif_path)
        xa_array_dict[filename] = tif_array.rename(filename)
        if target_resolution_d:
            tif_array = spatial_data.upsample_xarray_to_target(
                xa_array=tif_array, target_resolution=target_resolution_d
            )
        # save array to nc file
        file_ops.save_nc(tif_dir, filename, tif_array)

    return tif_paths
    print(f"All tifs converted to xarrays and stored as .nc files in {nc_dir}.")


# def process_reef_extent_tifs():
# fetch list of ground truth tifs
# gt_tif_files = file_ops.return_list_filepaths(
#     directories.get_gt_files_dir(), ".tif"
# )
# # generate dictionary of file names and arrays: {filename: xarray.DataArray, ...}
# gt_tif_dict = spatial_data.tifs_to_xa_array_dict(gt_tif_files)
# # save dictionary of tifs to nc, if files not already existing
# file_ops.save_dict_xa_ds_to_nc(gt_tif_dict, directories.get_gt_files_dir())


# utils/config.py
####################################################################################################
# entire file. Now used in config file, and guarantee_existence moved to file_ops.py


def get_coralshift_dir():
    coralshift_module = importlib.import_module("coralshift")
    coralshift_dir = Path(inspect.getabsfile(coralshift_module)).parent
    return (coralshift_dir / "..").resolve()


# utils/directories.py
####################################################################################################
# this was the format for the whole script, listing various different directories. Been superceded
# by config.py which does it without defining a load of existing files.


def get_volume_dir(volume_name: str) -> Path:
    """~/Volumes/volume_name"""
    return Path("/Volumes", volume_name)


# etc.


# data_structure.py
####################################################################################################
# this was the whole script. May be a useful format for thinking about remote/local work but is useless
# in its current form


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
        # elif location == "local":
        # getting the local/remote directory location
        # self.files_location = directories.get_volume_dir(volume_name) / "datasets/"
        # else:
        #     raise ValueError(f"Unrecognised location: {location}")

    def get_location(self):
        return self.files_location

    def add_dataset(self, name: str, data):
        self.datasets[name] = data

    def add_datasets(self, names: list[str], data: list):
        for i, name in enumerate(names):
            self.datasets[name] = data[i]

    def get_dataset(self, name: str):
        dataset = self.datasets.get(name, None)
        if dataset is None:
            raise ValueError(f"Dataset '{name}' does not exist.")
        return dataset

    def remove_dataset(self, name: str):
        if name in self.datasets:
            del self.datasets[name]

    def list_datasets(self):
        return list(self.datasets.keys())


# get_data.py
####################################################################################################


def preprocess_data(df, label_column):
    # Split dataframe into training and testing sets
    X_train, X_test, y_train, y_test = get_data.train_test_split(
        df.loc[:, df.columns != label_column],
        df[label_column],
        test_size=0.2,
        random_state=42,
    )

    # Min-max scaling ignoring NaNs
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train.drop(columns=[label_column]))
    X_test_scaled = scaler.transform(X_test.drop(columns=[label_column]))

    # Convert scaled arrays back to Dask DataFrame
    columns = X_train.drop(columns=[label_column]).columns
    X_train_scaled_df = dd.from_dask_array(
        pd.DataFrame(X_train_scaled, columns=columns)
    )
    X_test_scaled_df = dd.from_dask_array(pd.DataFrame(X_test_scaled, columns=columns))

    # Generate a column showing 1 if any NaN is in the row, 0 otherwise
    X_train_scaled_df["onehot_nan"] = X_train_scaled_df.isnull().any(axis=1).astype(int)
    X_test_scaled_df["onehot_nan"] = X_test_scaled_df.isnull().any(axis=1).astype(int)

    # Replace NaNs with 0
    X_train_scaled_df = X_train_scaled_df.fillna(0)
    X_test_scaled_df = X_test_scaled_df.fillna(0)

    return X_train_scaled_df, X_test_scaled_df


# I THINK THIS HAS BEEN REPLACED BY THE DATA CLASS DEFINTION BUT IT'S HERE IN CASE
def generate_gebco_xarray(
    resolution: float = 0.1,
    lats: list[float, float] = [-40, 0],
    lons: list[float, float] = [130, 170],
):
    # TODO: make generic for other bathymetry data?

    # check if gebco xarray already exists
    # res_str = utils.replace_dot_with_dash(str(resolution))

    gebco_xa_dir = Path(config.bathymetry_dir) / "gebco/rasters"
    gebco_xa_dir.mkdir(parents=True, exist_ok=True)

    # check if there exists a file spanning an adequate spatial extent
    nc_fps = list(Path(gebco_xa_dir).rglob("*.nc"))
    subset_fps = functions_creche.find_files_for_area(nc_fps, lats, lons)
    if len(subset_fps) > 0:
        # arbitrarily select first suitable file
        gebco_fp = subset_fps[0]
    else:
        raise FileNotFoundError("No GEBCO files with suitable geographic range found.")
    # TODO: proper chunking
    return xa.open_dataset(gebco_fp)


def generate_cmip_raster(
    variables: list[str],
    lats: list[float, float],
    lons: list[float, float],
    levs: list[int, int],
    year_range_to_include: list[int, int],
    source: str = "EC-Earth3P-HR",
    member: str = "r1i1p1f1",
):  # not currrently used

    get_data.ensure_cmip6_downloaded(variables, lats=lats, lons=lons, levs=levs)

    fname = functions_creche.FileName(
        variable_id=variables,
        grid_type="latlon",
        fname_type="var_concatted",
        lats=lats,
        lons=lons,
        plevels=levs,
        date_range=[min(year_range_to_include), max(year_range_to_include)],
    ).construct_fname()

    cmip_data_dir = (
        Path(config.cmip6_data_folder) / source / member / "testing"
    )  # TODO: replace testing
    cmip_fp = cmip_data_dir / fname

    return spatial_data.process_xa_d(xa.open_dataset(cmip_fp))


def get_ml_ready_static_data(
    res_str: str,
    depth_mask_lims: list[float, float],
    ground_truth_name: str,
    static_data_dir_fp: str,
    ttv_fractions: list[float, float, float],
    split_method: str,
    scale_method_X: str,
    scale_method_y: str,
    remove_nan_rows: bool,
    random_state: int = 42,
    verbosity: int = 0,
):
    # TODO: shift from global to arguments
    # load data. TODO: add multifile loading for machines with reasonable numbers of CPUs
    if verbosity > 0:
        print("loading static data...")
    all_data_df = get_data.load_static_data(res_str)

    # DATA PREPROCESSING
    # apply depth mask
    if verbosity > 0:
        print("preprocessing...")
    depth_masked_df = functions_creche.depth_filter(
        all_data_df, depth_mask_lims=depth_mask_lims
    )
    # split data into train, test, validation
    [train_df, test_df, val_df] = functions_creche.train_test_val_split(
        depth_masked_df,
        ttv_fractions=ttv_fractions,
        split_method=split_method,
        random_state=random_state,
    )
    # preprocess data: initialise scalers values
    scaler_X = get_data.generate_data_scaler(scale_method_X)
    scaler_y = get_data.generate_data_scaler(scale_method_y)
    # preprocess data: nan handling
    [train_df, test_df, val_df] = [
        get_data.onehot_nan(df, remove_nan_rows) for df in [train_df, test_df, val_df]
    ]
    # split into X and y for each subset
    (train_X, train_y), (test_X, test_y), (val_X, val_y) = [
        get_data.split_X_y(df, ground_truth_name) for df in [train_df, test_df, val_df]
    ]
    # fit scalers
    scaler_X.fit(train_X)
    scaler_y.fit(train_y)
    # transform data
    (train_X, train_y), (test_X, test_y), (val_X, val_y) = [
        (scaler_X.transform(X), scaler_y.transform(y))
        for X, y in [(train_X, train_y), (test_X, test_y), (val_X, val_y)]
    ]
    return (train_X, train_y), (test_X, test_y), (val_X, val_y)


# machine_learning/baselines.py
####################################################################################################


def generate_test_train_coordinates_multiple_areas(
    xa_ds_list: list[xa.Dataset],
    split_type: str = "pixel",
    test_lats: tuple[float] = None,
    test_lons: tuple[float] = None,
    test_fraction: float = 0.2,
    bath_mask: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate test and train coordinates for multiple areas.

    Args:
        xa_ds_list (list[xa.Dataset]): List of xarray datasets.
        split_type (str, optional): Split type. Defaults to "pixel".
        test_lats (tuple[float], optional): Latitude bounds for the test set. Defaults to None.
        test_lons (tuple[float], optional): Longitude bounds for the test set. Defaults to None.
        test_fraction (float, optional): Fraction of data to assign to the test set. Defaults to 0.2.
        bath_mask (bool, optional): Flag to apply a bathymetry mask. Defaults to True.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple of pandas DataFrames containing the train coordinates and test
            coordinates.
    """
    train_coords = pd.concat(
        (
            functions_creche.generate_test_train_coordinates(
                xa_ds, split_type, test_lats, test_lons, test_fraction, bath_mask
            )[0]
            for xa_ds in xa_ds_list
        ),
        ignore_index=True,
        axis=0,
    )
    test_coords = pd.concat(
        (
            functions_creche.generate_test_train_coordinates(
                xa_ds, split_type, test_lats, test_lons, test_fraction, bath_mask
            )[1]
            for xa_ds in xa_ds_list
        ),
        ignore_index=True,
        axis=0,
    )

    return train_coords, test_coords


# not really sure what doing
def vary_train_test_lat(
    df: pd.DataFrame,
    num_vals: int = 10,
    train_direction: str = "N",
    random_seed: int = 42,
) -> list[tuple[float, float]]:
    lat_min, lat_max = df.index["latitude"].min(), df.index["latitude"].max()
    lat_dividers = np.linspace(lat_min, lat_max, num_vals)

    train_coords_list, test_coords_list = [], []
    for lat_divide in lat_dividers:
        train_coords, test_coords = functions_creche.generate_test_train_coords_from_df(
            df=df,
            split_type="spatial",
            train_test_lat_divide=lat_divide,
            train_direction=train_direction,
            random_seed=random_seed,
        )
        train_coords_list.append(train_coords)
        test_coords_list.append(test_coords)
    return lat_divide, train_coords_list, test_coords_list


# not really sure what doing
def generate_test_train_coords_from_dfs(
    dfs: list[pd.DataFrame],
    test_fraction: float = 0.25,
    split_type: str = "pixel",
    train_test_lat_divide: int = -18,
    train_direction: str = "N",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_coords_list, test_coords_list = [], []
    for df in dfs:
        lists = functions_creche.generate_test_train_coords_from_df(
            df=df,
            test_fraction=test_fraction,
            split_type=split_type,
            train_test_lat_divide=train_test_lat_divide,
            train_direction=train_direction,
        )
        train_coords_list.append(lists[0])
        test_coords_list.append(lists[1])

    return train_coords_list, test_coords_list


# too non-descriptive and particular
def process_df_for_ml(
    df: pd.DataFrame, ignore_vars: list[str], drop_all_nans: bool = True
) -> pd.DataFrame:
    # drop ignored vars
    df = df.drop(columns=list(set(ignore_vars).intersection(df.columns)))

    if drop_all_nans:
        # remove rows which are all nans
        df = utils.drop_nan_rows(df)
    # onehot encoode any remaining nans
    df["onehotnan"] = df.isnull().any(axis=1).astype(int)
    # fill nans with 0
    df = df.fillna(0)

    # flatten dataset for row indexing and model training
    return df


# too much non-transparently going on in this. If anything, should split into several. But not worth it
def xa_dss_to_df(
    xa_dss: list[xa.Dataset],
    bath_mask: bool = True,
    ignore_vars: list = ["spatial_ref", "band", "depth"],
    drop_all_nans: bool = True,
):
    dfs = []
    for xa_ds in xa_dss:
        if bath_mask:
            # set all values outside of the shallow water region to nan for future omission
            shallow_mask = spatial_data.generate_var_mask(xa_ds)
            xa_ds = xa_ds.where(shallow_mask, np.nan)

        # compute out dasked chunks, send type to float32, stack into df, drop any datetime columns
        df = (
            xa_ds.stack(points=("latitude", "longitude"))
            .compute()
            .astype("float32")
            .to_dataframe()
        )
        # drop temporal columns
        df = df.drop(columns=list(df.select_dtypes(include="datetime64").columns))
        df = process_df_for_ml(df, ignore_vars=ignore_vars, drop_all_nans=drop_all_nans)

        dfs.append(df)
    return dfs


# got to be a less janky way to do this. I've grown up since then...
def visualise_train_test_split(xa_ds: xa.Dataset, train_coordinates, test_coordinates):
    """
    Visualizes the training and testing regions on a spatial grid.

    Parameters:
        xa_ds (xarray.Dataset): Input dataset containing spatial grid information.
        train_coordinates (list): List of training coordinates as (latitude, longitude) tuples.
        test_coordinates (list): List of testing coordinates as (latitude, longitude) tuples.

    Returns:
        xarray.DataArray: DataArray with the same coordinates and dimensions as xa_ds, where the spatial pixels
                          corresponding to training and testing regions are color-coded.
    """
    lat_spacing = xa_ds.latitude.values[1] - xa_ds.latitude.values[0]
    lon_spacing = xa_ds.longitude.values[1] - xa_ds.longitude.values[0]

    array_shape = tuple(xa_ds.dims[d] for d in list(xa_ds.dims))
    train_pixs, test_pixs = np.empty(array_shape), np.empty(array_shape)
    train_pixs[:] = np.nan
    test_pixs[:] = np.nan
    # Color the spatial pixels corresponding to training and testing regions
    for train_index in tqdm(train_coordinates, desc="Coloring in training indices..."):
        row, col = spatial_data.find_coord_indices(
            xa_ds, train_index[0], train_index[1], lat_spacing, lon_spacing
        )
        train_pixs[row, col] = 0

    for test_index in tqdm(test_coordinates, desc="Coloring in training indices..."):
        row, col = spatial_data.find_coord_indices(
            xa_ds, test_index[0], test_index[1], lat_spacing, lon_spacing
        )
        test_pixs[row, col] = 1

    train_test_ds = xa.DataArray(
        np.nansum(np.stack((train_pixs, test_pixs)), axis=0),
        coords=xa_ds.coords,
        dims=xa_ds.dims,
    )
    return train_test_ds


# I can't imagine I care that much about overwriting (static) models any more, especially since I save parameters
# elsewhere
def save_sklearn_model(model, savedir: Path | str, filename: str) -> None:
    """
    Save a scikit-learn model to a file using pickle.

    Parameters
    ----------
    model : object
        The scikit-learn model object to be saved.
    savedir : Union[pathlib.Path, str]
        The directory path where the model file should be saved.
    filename : str
        The name of the model file.

    Returns
    -------
    None
    """
    save_path = (Path(savedir) / filename).with_suffix(".pickle")
    if save_path.is_file():
        save_path = (Path(savedir) / f"{filename}_new").with_suffix(".pickle")
        print(f"Model at {save_path} already exists.")
    # save model
    with open(save_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved model to {save_path}.")
    return save_path


# # this is just gross – how ungeneralisable and random can you be? A niche application of returning a metrics dict
# def evaluate_model(y_test: np.ndarray | pd.Series, predictions: np.ndarray):
#     """
#     Evaluate a model's performance using regression and classification metrics.

#     Parameters
#     ----------
#         y_test (np.ndarray or pd.Series): True labels.
#         predictions (np.ndarray or pd.Series): Predicted labels.

#     Returns
#     -------
#         tuple[float, float]: Tuple containing the mean squared error (regression metric) and binary cross-entropy
#             (classification metric).
#     """
#     # calculate regression (mean-squared error) metric
#     mse = sklmetrics.mean_squared_error(y_test, predictions)

#     # calculate classification (binary cross-entropy/log_loss) metric
#     y_thresh, y_pred_thresh = threshold_label(y_test, predictions)
#     bce = sklmetrics.log_loss(y_thresh, y_pred_thresh)

#     return mse, bce


# old, disgustingly inefficient way to calculate the Couce+ metrics
# def generate_reproducing_metrics(
#     resampled_xa_das_dict: dict, target_resolution_d: float = None, region: str = None
# ) -> xa.Dataset:
#     """
#     Generate metrics used in Couce et al (2013, 2023) based on the upsampled xarray DataArrays.

#     Parameters
#     ----------
#         resampled_xa_das_dict (dict): A dictionary containing the upsampled xarray DataArrays.

#     Returns
#     -------
#         xa.Dataset: An xarray Dataset containing the reproduced metrics.
#     """
#     if target_resolution_d:
#         resolution = target_resolution_d
#     else:
#         resolution = np.mean(
#             spatial_data.calculate_spatial_resolution(resampled_xa_das_dict["thetao"])
#         )

#     res_string = utils.replace_dot_with_dash(f"{resolution:.04f}d")

#     # if not region:
#     #     filename = f"{res_string}_arrays/all_{res_string}_comparative"
#     # else:
#     #     region_dir = file_ops.guarantee_existence(
#     #         directories.get_comparison_dir()
#     #         / f"{bathymetry.ReefAreas().get_short_filename(region)}/{res_string}_arrays"
#     #     )
#     #     # region_letter = bathymetry.ReefAreas().get_letter(region)
#     #     filename = region_dir / f"all_{res_string}_comparative"

#     # save_path = (directories.get_comparison_dir() / filename).with_suffix(".nc")

#     if not save_path.is_file():
#         # TEMPERATURE
#         thetao_daily = resampled_xa_das_dict["thetao"]
#         # annual average, stdev of annual averages, annual minimum, annual maximum
#         (
#             thetao_annual_average,
#             _,
#             (thetao_annual_min, thetao_annual_max),
#         ) = calc_timeseries_params(thetao_daily, "y", "thetao")
#         # monthly average, stdev of monthly averages, monthly minimum, monthly maximum
#         (
#             thetao_monthly_average,
#             thetao_monthly_stdev,
#             (thetao_monthly_min, thetao_monthly_max),
#         ) = calc_timeseries_params(thetao_daily, "m", "thetao")
#         # annual range (monthly max - monthly min)
#         thetao_annual_range = (thetao_annual_max - thetao_annual_min).rename(
#             "thetao_annual_range"
#         )
#         # weekly minimum, weekly maximum
#         _, _, (thetao_weekly_min, thetao_weekly_max) = calc_timeseries_params(
#             thetao_daily, "w", "thetao"
#         )
#         print("Generated thetao data")

#         # SALINITY
#         salinity_daily = resampled_xa_das_dict["so"]
#         # annual average
#         salinity_annual_average, _, _ = calc_timeseries_params(
#             salinity_daily, "y", "salinity"
#         )
#         # monthly min, monthly max
#         (
#             _,
#             _,
#             (salinity_monthly_min, salinity_monthly_max),
#         ) = calc_timeseries_params(salinity_daily, "m", "salinity")
#         print("Generated so data")

#         # CURRENT
#         current_daily = resampled_xa_das_dict["current"]
#         # annual average
#         current_annual_average, _, _ = calc_timeseries_params(
#             current_daily, "y", "current"
#         )
#         # monthly min, monthly max
#         (
#             _,
#             _,
#             (current_monthly_min, current_monthly_max),
#         ) = calc_timeseries_params(current_daily, "m", "current")
#         print("Generated current data")

#         # BATHYMETRY
#         bathymetry_climate_res = resampled_xa_das_dict["bathymetry"]
#         print("Generated bathymetry data")

#         # # ERA5
#         solar_daily = resampled_xa_das_dict["ssr"]
#         # annual average
#         solar_annual_average, _, _ = calc_timeseries_params(
#             solar_daily, "y", "net_solar"
#         )
#         # monthly min, monthly max
#         _, _, (solar_monthly_min, solar_monthly_max) = calc_timeseries_params(
#             solar_daily, "m", "net_solar"
#         )
#         print("Generated solar data")

#         # GT
#         gt_climate_res = resampled_xa_das_dict["gt"]
#         print("Generated ground truth data")

#         merge_list = [
#             thetao_annual_average.mean(dim="time"),
#             thetao_annual_range,
#             thetao_monthly_min,
#             thetao_monthly_max,
#             thetao_monthly_stdev,
#             thetao_weekly_min,
#             thetao_weekly_max,
#             salinity_annual_average.mean(dim="time"),
#             salinity_monthly_min,
#             salinity_monthly_max,
#             current_annual_average.mean(dim="time"),
#             current_monthly_min,
#             current_monthly_max,
#             solar_annual_average.mean(dim="time"),
#             solar_monthly_min,
#             solar_monthly_max,
#             gt_climate_res,
#             bathymetry_climate_res,
#         ]
#         for xa_da in merge_list:
#             if "grid_mapping" in xa_da.attrs:
#                 del xa_da.attrs["grid_mapping"]
#         # MERGE
#         merged = xa.merge(merge_list).astype(np.float64)
#         merged.attrs["region"] = region
#         with np.errstate(divide="ignore", invalid="ignore"):
#             merged.to_netcdf(save_path)
#             return merged

#     else:
#         print(f"{save_path} already exists.")
#         return xa.open_dataset(save_path, decode_coords="all")


# dependencies for the ridiculous function above
def calc_timeseries_params(xa_da_daily_means: xa.DataArray, period: str, param: str):
    """
    Calculates time series parameters (mean, standard deviation, min, max) for a given period.

    Parameters
    ----------
    xa_da_daily_means (xarray.DataArray): The input xarray DataArray of daily means.
    period (str): The time period for grouping (e.g., 'year', 'month', 'week').
    param (str): The parameter name.

    Returns
    -------
    xarray.DataArray, xarray.DataArray, tuple(xarray.DataArray, xarray.DataArray): The weighted average,
        standard deviation, and (min, max) values for the given period.
    """
    # determine original crs
    crs = xa_da_daily_means.rio.crs

    base_name = f"{param}_{period}_"
    # weighted average
    weighted_av = calc_time_weighted_mean(xa_da_daily_means, period)

    weighted_av = weighted_av.rename(base_name + "mean")
    # standard deviation of weighted averages
    stdev = (
        weighted_av.std(dim="time", skipna=True)
        .rename(base_name + "std")
        .rio.write_crs(crs, inplace=True)
    )
    # max and min
    min_val = (
        xa_da_daily_means.min(dim="time", skipna=True)
        .rename(base_name + "min")
        .rio.write_crs(crs, inplace=True)
    )
    max_val = (
        xa_da_daily_means.max(dim="time", skipna=True)
        .rename(base_name + "max")
        .rio.write_crs(crs, inplace=True)
    )

    # Return the weighted average
    return weighted_av, stdev, (min_val, max_val)


def return_time_grouping_offset(period: str):
    """
    Returns the time grouping and offset for a given period.

    Parameters
    ----------
    period (str): The time period for grouping (e.g., 'year', 'month', 'week').

    Returns
    -------
    group (str): The time grouping for the given period.
    offset (str): The offset for the given period.
    """
    if period.lower() in ["year", "y", "annual"]:
        group = "time.year"
        offset = "AS"
    elif period.lower() in ["month", "m"]:
        group = "time.month"
        offset = "MS"
    elif period.lower() in ["week", "w"]:
        group = "time.week"
        offset = pd.offsets.Week()

    return group, offset


def calc_time_weighted_mean(xa_da_daily_means: xa.DataArray, period: str):
    """
    Calculates the weighted mean of daily means for a given period.

    Parameters
    ----------
    xa_da_daily_means (xarray.DataArray): The input xarray DataArray of daily means.
    period (str): The time period for grouping (e.g., 'year', 'month', 'week').

    Returns
    -------
    xarray.DataArray: The weighted mean values for the given period.
    """
    group, offset = return_time_grouping_offset(period)
    # determine original crs
    crs = xa_da_daily_means.rio.crs

    # Determine the month length (has no effect on other time periods)
    month_length = xa_da_daily_means.time.dt.days_in_month
    # Calculate the monthly weights
    weights = month_length.groupby(group) / month_length.groupby(group).sum()

    # Setup our masking for nan values
    ones = xa.where(xa_da_daily_means.isnull(), 0.0, 1.0)

    # Calculate the numerator
    xa_da_daily_means_sum = (
        (xa_da_daily_means * weights).resample(time=offset).sum(dim="time")
    )
    # Calculate the denominator
    ones_out = (ones * weights).resample(time=offset).sum(dim="time")

    # weighted average
    return (xa_da_daily_means_sum / ones_out).rio.write_crs(crs, inplace=True)


# overkill, used in conjunction with calc_time_weighted_mean I believe
def calc_weighted_mean(xa_da_daily_means: xa.DataArray, period: str):
    """
    Calculates the weighted mean of daily means for a given period.

    Parameters
    ----------
    xa_da_daily_means (xarray.DataArray): The input xarray DataArray of daily means.
    period (str): The time period for grouping (e.g., 'year', 'month', 'week').

    Returns
    -------
    xarray.DataArray: The weighted mean values for the given period.
    """
    group, offset = return_time_grouping_offset(period)
    # determine original crs
    crs = xa_da_daily_means.rio.crs

    # Determine the month length (has no effect on other time periods)
    month_length = xa_da_daily_means.time.dt.days_in_month
    # Calculate the monthly weights
    weights = month_length.groupby(group) / month_length.groupby(group).sum()

    # Setup our masking for nan values
    ones = xa.where(xa_da_daily_means.isnull(), 0.0, 1.0)

    # Calculate the numerator
    xa_da_daily_means_sum = (
        (xa_da_daily_means * weights).resample(time=offset).sum(dim="time")
    )
    # Calculate the denominator
    ones_out = (ones * weights).resample(time=offset).sum(dim="time")

    # weighted average
    return (xa_da_daily_means_sum / ones_out).rio.write_crs(crs, inplace=True)


def calculate_class_weight(label_array: np.ndarray):
    unique_values, counts = np.unique(label_array, return_counts=True)
    occurrence_dict = dict(zip(unique_values, counts))
    return occurrence_dict


# straight-up duplicate from functions_creche
def threshold_array(
    array: np.ndarray | pd.Series, threshold: float = 0.25
) -> pd.Series:
    result = np.where(np.array(array) > threshold, 1, 0)
    if isinstance(array, pd.Series):
        return pd.Series(result, index=array.index)
    else:
        return pd.Series(result)


# straight-up duplicate from functions_creche
def threshold_label(
    labels: np.ndarray | pd.Series,
    predictions: np.ndarray | pd.Series,
    threshold: float,
) -> tuple[np.ndarray]:
    """Apply thresholding to labels and predictions.

    Parameters
    ----------
        labels (np.ndarray or pd.Series): True labels.
        predictions (np.ndarray or pd.Series): Predicted labels.
        threshold (float): Threshold value for binary classification.

    Returns
    -------
        tuple[np.ndarray]: Tuple containing thresholded labels and thresholded predictions.
    """
    thresholded_labels = threshold_array(labels, threshold)
    thresholded_preds = threshold_array(predictions, threshold)
    return thresholded_labels, thresholded_preds


# A mere shadow of what I've achieved with Classes
def create_train_metadata(
    name: str,
    model_path: Path | str,
    model_type: str,
    data_type: str,
    search_type: str,
    fit_time: float,
    test_fraction: float,
    coord_ranges: dict,
    features: list[str],
    resolution: float,
    n_iter: int,
    cv: int,
    param_distributions: dict,
    random_state: int,
    n_jobs: int,
) -> None:
    metadata = {
        "model name": name,
        "model path": str(model_path),
        "model type": model_type,
        "data type": data_type,
        "search type": search_type,
        "model fit time (s)": fit_time,
        "test fraction": test_fraction,
        "features": features,
        "approximate spatial resolution": resolution,
        "number of fit iterations (random search)": n_iter,
        "cross validation fold size": cv,
        "random_state": 42,
        "n_jobs": -1,
        "latitude/longitude limits of testing data (all else used in training)": " ",
    }
    metadata.update(coord_ranges)
    # update metadata with parameter search grid
    metadata.update(param_distributions)

    filename = f"{name}_metadata"
    save_path = (Path(model_path).parent / filename).with_suffix(".json")
    if save_path.is_file():
        save_path = (Path(model_path).parent / f"{filename}_new").with_suffix(".json")
        print(f"Metadata at {save_path} already exists.")
    out_file = open(save_path, "w")
    json.dump(metadata, out_file, indent=4)
    print(f"{name} metadata saved to {save_path}")


# # have expanded this hugely in functions_creche.py
# class ModelInitialiser:
#     def __init__(self, random_state: int = 42):
#         self.random_state = random_state

#         self.model_info = [
#             # continuous model
#             {
#                 "model_type": "rf_reg",
#                 "data_type": "continuous",
#                 "model": RandomForestRegressor(
#                     verbose=1, random_state=self.random_state
#                 ),
#                 "search_grid": rf_search_grid(),
#             },
#             {
#                 "model_type": "brt",
#                 "data_type": "continuous",
#                 "model": GradientBoostingRegressor(
#                     verbose=1, random_state=self.random_state
#                 ),
#                 "search_grid": boosted_regression_search_grid(),
#             },
#             # discrete models
#             {
#                 "model_type": "rf_cla",
#                 "data_type": "discrete",
#                 "model": RandomForestClassifier(
#                     class_weight="balanced", verbose=1, random_state=self.random_state
#                 ),
#                 "search_grid": rf_search_grid(),
#             },
#             {
#                 "model_type": "maxent",
#                 "data_type": "discrete",
#                 "model": LogisticRegression(
#                     class_weight="balanced", verbose=1, random_state=self.random_state
#                 ),
#                 "search_grid": maximum_entropy_search_grid(),
#             },
#         ]

#     def get_data_type(self, model_type):
#         for m in self.model_info:
#             if m["model_type"] == model_type:
#                 return m["data_type"]
#         else:
#             raise ValueError(f"'{model_type}' not a valid model.")

#     def get_model(self, model_type):
#         for m in self.model_info:
#             if m["model_type"] == model_type:
#                 return m["model"]
#         else:
#             raise ValueError(f"'{model_type}' not a valid model.")

#     def get_search_grid(self, model_type):
#         for m in self.model_info:
#             if m["model_type"] == model_type:
#                 return m["search_grid"]
#         else:
#             raise ValueError(f"'{model_type}' not a valid model.")


# # weirdly specific outputs, not useful
# def initialise_model(model_type: str, random_state: int = 42):
#     model_instance = ModelInitialiser(random_state=random_state)

#     return (
#         model_instance.get_model(model_type),
#         model_instance.get_data_type(model_type),
#     )


# adding very little vs doing it within the model class/function call
# def initialise_grid_search(
#     model_type, best_params_dict, cv: int = 3, num_samples: int = 3, verbose=0
# ):
#     param_grid = generate_parameter_grid(best_params_dict)
#     # generate_gridsearch_grid(best_params_dict)
#     model_class = ModelInitialiser()
#     model = model_class.get_model(model_type)

#     return (
#         GridSearchCV(
#             estimator=model, param_grid=param_grid, cv=cv, n_jobs=-1, verbose=verbose
#         ),
#         param_grid,
#     )


# # superceded by class system
# def train_tune(
#     X_train: pd.DataFrame,
#     y_train: np.ndarray | pd.Series,
#     model_type: str,
#     # resolution: float,
#     name: str = "_",
#     # test_fraction: float = 0.25,
#     save_dir: Path | str = None,
#     n_iter: int = 50,
#     cv: int = 3,
#     num_samples: int = 3,
#     search_type: str = "random",
#     best_params_dict: dict = None,
#     n_jobs: int = -1,
#     verbose: int = 0,
# ):
#     model, data_type = initialise_model(model_type)

#     if data_type == "discrete":
#         y_train = threshold_array(y_train)

#     if search_type == "random":
#         search_grid = ModelInitialiser().get_search_grid(model_type)
#         model_search = RandomizedSearchCV(
#             estimator=model,
#             param_distributions=search_grid,
#             n_iter=n_iter,
#             cv=cv,
#             verbose=verbose,
#             random_state=42,
#             n_jobs=n_jobs,
#         )
#         print("Fitting model with a randomized hyperparameter search...")
#     elif search_type == "grid":
#         model_search, search_grid = initialise_grid_search(
#             model_type,
#             best_params_dict,
#             cv=cv,
#             num_samples=num_samples,
#             verbose=verbose,
#         )
#         print("Fitting model with a grid hyperparameter search...")
#     else:
#         raise ValueError(f"Search type: {search_type} not recognised.")

#     start_time = time.time()
#     model_search.fit(X_train, y_train)
#     end_time = time.time()
#     fit_time = end_time - start_time

#     return model_search


# # superspecific chain of building blocks, old bathymetry system
# def load_and_process_reproducing_xa_das(
#     region: str, chunk_dict: dict = {"latitude": 100, "longitude": 100, "time": 100}
# ) -> list[xa.DataArray]:
#     """
#     Load and process xarray data arrays for reproducing metrics.

#     Returns
#     -------
#         list[xa.DataArray]: A list containing the processed xarray data arrays.
#     """
#     region_name = bathymetry.ReefAreas().get_short_filename(region)
#     region_letter = bathymetry.ReefAreas().get_letter(region)

#     # load in daily sea water potential temp
#     # thetao_daily = xa.open_dataarray(directories.get_processed_dir() / "arrays/thetao.nc")

#     dailies_array = xa.open_dataset(
#         directories.get_daily_cmems_dir()
#         / f"{region_name}/cmems_gopr_daily_{region_letter}.nc",
#         decode_coords="all",
#         chunks=chunk_dict,
#     ).isel(depth=0)

#     # load in daily sea water potential temp
#     thetao_daily = dailies_array["thetao"]
#     # load in daily sea water salinity means
#     salinity_daily = dailies_array["so"]
#     # calculate current magnitude
#     current_daily = calculate_magnitude(
#         dailies_array["uo"].compute(), dailies_array["vo"].compute()
#     ).rename("current")
#     # TODO: download additional ERA5 files
#     # load bathymetry file TODO: separate function for this
#     bath_file = list(
#         directories.get_bathymetry_datasets_dir().glob(f"{region_name}_*.nc")
#     )[0]
#     bath = xa.open_dataarray(bath_file, decode_coords="all", chunks=chunk_dict).rename(
#         "bathymetry"
#     )
#     # Load in ERA5 surface net solar radiation
#     net_solar_file = list(
#         (directories.get_era5_data_dir() / f"{region_name}/weather_parameters/").glob(
#             "*surface_net_solar_radiation_*.nc"
#         )
#     )[0]
#     net_solar = spatial_data.process_xa_d(
#         xa.open_dataarray(net_solar_file, decode_coords="all", chunks=chunk_dict)
#     )
#     net_solar = net_solar.resample(time="1D").mean(dim="time")

#     # Load in ground truth coral data
#     gt = xa.open_dataarray(
#         directories.get_gt_files_dir() / f"coral_region_{region_letter}_1000m.nc",
#         decode_coords="all",
#         chunks=chunk_dict,
#     ).rename("gt")

#     return [thetao_daily, salinity_daily, current_daily, net_solar, bath, gt]


# # specificity makes for unnecessary complexity
# def load_model(
#     model_name: Path | str, model_dir: Path | str = directories.get_best_model_dir()
# ):
#     if Path(model_name).is_file():
#         model = pickle.load(open(model_name, "rb"))
#     else:
#         model = pickle.load(open(Path(model_dir) / model_name, "rb"))
#     print(model.best_params_)
#     return model


# # bathymetry-specific, and not in a good, generalisable way. Should first create a collective raster
# # of the areas, and use this as underlay
# def generate_reproducing_metrics_for_regions(
#     regions_list: list = ["A", "B", "C", "D"], target_resolution_d: float = 1 / 27
# ) -> None:
#     for region in tqdm(
#         regions_list,
#         total=len(regions_list),
#         position=0,
#         leave=False,
#         desc=" Processing regions",
#     ):
#         lat_lims = bathymetry.ReefAreas().get_lat_lon_limits(region)[0]
#         lon_lims = bathymetry.ReefAreas().get_lat_lon_limits(region)[1]

#         # create list of xarray dataarrays
#         reproduction_xa_list = load_and_process_reproducing_xa_das(region)
#         # create dictionary of xa arrays, resampled to correct resolution
#         resampled_xa_das_dict = file_ops.resample_list_xa_ds_into_dict(
#             reproduction_xa_list,
#             target_resolution=target_resolution_d,
#             unit="d",
#             lat_lims=lat_lims,
#             lon_lims=lon_lims,
#         )
#         # generate and save reproducing metrics from merged dict
#         generate_reproducing_metrics(
#             resampled_xa_das_dict,
#             region=region,
#             target_resolution_d=target_resolution_d,
#         )


# too specific
def outputs_to_xa_ds(labels, predictions) -> xa.Dataset:
    if type(predictions) is not np.ndarray:
        predictions = predictions.to_numpy()
    df = pd.DataFrame({"labels": labels, "predictions": predictions})
    return df.to_xarray().sortby(["longitude", "latitude"])


# ew! so many random arguments. Plotting,
def evaluate_model(
    model, df: pd.DataFrame, X: np.ndarray, y: np.ndarray, figsize: tuple = [4, 4]
) -> None:
    """
    Evaluate model (visually and mse) on a given dataset, returning an xarray with predictions and ground truth.

    Args:
        model (sklearn model): trained model
        df (pd.DataFrame): dataframe with ground truth
        X (np.ndarray): input data
        y (np.ndarray): ground truth
        figsize (tuple, optional): figure size. Defaults to [4,4].

    Returns:
        pred_xa (xa.Dataset): xarray dataset with ground truth and predictions
    """
    y_pred = model.predict(X)
    pred_df = functions_creche.reform_df(df, y_pred)
    mse = sklmetrics.mean_squared_error(
        pred_df["unep_coral_presence"], pred_df["prediction"]
    )

    f, ax = plt.subplots(figsize=figsize)
    ax.scatter(y, y_pred)
    # y=x for comparison
    ax.axline((0, 0), slope=1, c="k")
    ax.axis("equal")
    ax.set_xlabel("Ground Truth")
    ax.set_ylabel("Prediction")
    ax.set_xlim([0, 1])

    plt.suptitle(f"MSE: {mse:.04f}")


# # seriously?? why would these be chained together??
# def generate_reproducing_metrics_at_different_resolutions(
#     resolutions: list[float], units: list[str]
# ) -> xa.Dataset:
#     target_resolutions = [
#         spatial_data.choose_resolution(number, string)[1]
#         for number, string in zip(resolutions, units)
#     ]
#     for res in tqdm(
#         target_resolutions,
#         total=len(target_resolutions),
#         desc="Processing metrics at various resolutions",
#     ):
#         files_dir = directories.get_comparison_dir() / utils.replace_dot_with_dash(
#             f"{res:.05f}d_arrays"
#         )
#         files = file_ops.return_list_filepaths(files_dir, ".nc")
#         xa_ds_dict = {}

#         for fl in files:
#             name = (fl.stem).split("_")[0]
#             ds = xa.open_dataset(fl).rio.write_crs("epsg:4326")
#             variable_name = next(
#                 (var_name for var_name in ds.data_vars if var_name != "spatial_ref"),
#                 None,
#             )
#             xa_ds_dict[name] = ds[variable_name]

#         generate_reproducing_metrics(xa_ds_dict, res)


# too specific to MRes work
# def get_comparison_xa_ds(
#     region_list: list = ["A", "B", "C", "D"], d_resolution: float = 0.0368
# ):
#     res_string = utils.replace_dot_with_dash(f"{d_resolution:.04f}d")

#     xa_dss = []
#     for region in region_list:
#         region_name = bathymetry.ReefAreas().get_short_filename(region)
#         all_data_dir = (
#             directories.get_comparison_dir() / f"{region_name}/{res_string}_arrays"
#         )
#         all_data_name = f"all_{res_string}_comparative"
#         xa_dss.append(
#             xa.open_dataset(
#                 (all_data_dir / all_data_name).with_suffix(".nc"), decode_coords="all"
#             )
#         )

#     return xa_dss

# # this would now be handled in conjunction with the class system/config files
# def train_tune_across_models(
#     model_types: list[str],
#     d_resolution: float = 0.03691,
#     split_type: str = "pixel",
#     test_lats: tuple[float] = None,
#     test_lons: tuple[float] = None,
#     test_fraction: float = 0.25,
#     cv: int = 3,
#     n_iter: int = 10,
# ):
#     model_comp_dir = file_ops.guarantee_existence(
#         directories.get_datasets_dir() / "model_params/best_models"
#     )

#     all_data = get_comparison_xa_ds(d_resolution=d_resolution)
#     res_string = utils.replace_dot_with_dash(f"{d_resolution:.04f}d")

#     # define train/test split so it's the same for all models
#     (X_trains, X_tests, y_trains, y_tests, _, _) = spatial_split_train_test(
#         all_data,
#         "gt",
#         split_type=split_type,
#         test_fraction=test_fraction,
#     )

#     for model in tqdm(
#         model_types, total=len(model_types), desc="Fitting each model via random search"
#     ):
#         train_tune(
#             X_train=X_trains,
#             y_train=y_trains,
#             model_type=model,
#             resolution=d_resolution,
#             save_dir=model_comp_dir,
#             name=f"{model}_{res_string}_tuned",
#             test_fraction=test_fraction,
#             cv=cv,
#             n_iter=n_iter,
#         )


""" Unused/superceded from xgb_functs.py """


def generate_data_scaler(method: str = "minmax"):
    """
    Scale the data using the specified method.
    """
    if method == "minmax":
        return MinMaxScaler()
    elif method == "standard":
        return StandardScaler()
    elif method == "log":
        return FunctionTransformer(log_transform)


# decorator to hold function output in memory
@lru_cache(maxsize=None)  # Set maxsize to limit the cache size, or None for unlimited
def load_static_data(res_str: str = "0-01"):
    # TODO: could replace with path to multifile dir, open with open_mfdataset and chunking
    high_res_ds_fp = config.static_cmip6_data_dir.glob(f"*_{res_str}_*.nc").__next__()
    high_res_ds = xa.open_dataset(high_res_ds_fp)
    all_data_df = high_res_ds.to_dataframe()

    return all_data_df


def split_X_y(df, target_name: str):
    y = df[target_name]
    X = df.drop(target_name, axis=1)

    return X, y


def generate_data_scaler(method: str = "minmax"):
    """
    Scale the data using the specified method.
    """
    return {
        "minmax": MinMaxScaler(),
        "standard": StandardScaler(),
        "robust": RobustScaler(),
        "quantile": QuantileTransformer(),
        "box-cox": PowerTransformer(method="box-cox"),
        "yeo-johnson": PowerTransformer(method="yeo-johnson"),
        "log": FunctionTransformer(log_transform),
    }.get(method)


def onehot_nan(df, discard_nanrows: bool = True):
    """
    One-hot encode the nan values in the data.

    TODO: should I just remove these rows outright?
    """
    nan_rows = df.isna().any(axis=1)

    if discard_nanrows:
        return df.dropna()
    else:
        # encode any rows containing nans to additional column
        df["nan_onehot"] = nan_rows.astype(int)
        # fill any nans with zeros
        return df.fillna(0)


# all CMIP commands below have been superceded by CMIPPER
def construct_cmip_command(
    variable: str,
    source_id: str = "EC-Earth3P-HR",
    member_id: str = "r1i1p2f1",
    lats: list[float, float] = [-40, 0],
    lons: list[float, float] = [130, 170],
    levs: list[int, int] = [0, 20],
    script_fp: str = "coralshift/dataloading/download_cmip6_data_parallel.py",
):
    arguments = [
        "--source_id",
        source_id,
        "--member_id",
        member_id,
        "--variable_id",
        variable,
    ]

    # Define the list arguments with their corresponding values
    list_arguments = [
        ("--lats", lats),
        ("--lons", lons),
        ("--levs", levs),
    ]
    list_str = [
        item
        for sublist in [
            [flag, str(val)] for flag, vals in list_arguments for val in vals
        ]
        for item in sublist
    ]

    arguments += list_str
    return ["/home/rt582/miniforge3/envs/coralshift/bin/python", script_fp] + arguments


def execute_subprocess_command(command, output_log_path):
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        print(f"{output_log_path}")
        with open(output_log_path, "w") as output_log_file:
            output_log_file.write(result.stdout.decode())
            output_log_file.write(result.stderr.decode())

        # with open(str(output_log_path), "w") as error_log_file:
        #     error_log_file.write(result.stderr.decode())

    except subprocess.CalledProcessError as e:
        print(f"Error{e}")
        # # Handle the exception if needed
        with open(output_log_path, "w") as error_log_file:
            error_log_file.write(f"Error: {e}")
            error_log_file.write(result.stderr.decode())


def run_commands_with_thread_pool(cmip_commands, output_log_paths, error_log_paths):
    with ThreadPoolExecutor(max_workers=len(cmip_commands)) as executor:
        # Submit each script command to the executor
        executor.map(execute_subprocess_command, cmip_commands, output_log_paths)


def ensure_cmip6_downloaded(
    variables: list[str],
    source_id: str = "EC-Earth3P-HR",
    member_id: str = "r1i1p2f1",
    lats: list[float, float] = [-40, 0],
    lons: list[float, float] = [130, 170],
    year_range_to_include: list[int, int] = [1950, 2014],
    levs: list[int, int] = [0, 20],
    logging_dir: str = config.logs_dir,
):
    potential_ds, potential_ds_fp = find_intersecting_cmip(
        variables, source_id, member_id, lats, lons, levs
    )
    if potential_ds is None:
        if not Path(logging_dir).exists():
            Path(logging_dir).mkdir(parents=True, exist_ok=True)

        print("Creating/downloading necessary file(s)...")
        # TODO: add in other sources/members

        cmip_commands, output_log_paths = construct_cmip_commands(
            variables, source_id, member_id, lats, lons, levs
        )
        run_commands_with_thread_pool(cmip_commands, output_log_paths, output_log_paths)
    else:
        print(
            f"CMIP6 file with necessary variables spanning latitudes {lats} and longitudes {lons} already exists at: \n{potential_ds_fp}"  # noqa
        )


def find_intersecting_cmip(
    variables: list[str],
    source_id: str = "EC-Earth3P-HR",
    member_id: str = "r1i1p2f1",
    lats: list[float, float] = [-40, 0],
    lons: list[float, float] = [130, 170],
    year_range: list[int, int] = [1950, 2014],
    levs: list[int, int] = [0, 20],
):

    # check whether intersecting cropped file already exists
    cmip6_dir_fp = (
        config.cmip6_data_dir / source_id / member_id / "testing" / "regridded"
    )
    # TODO: fix bas path: maps-priv at start is screwing everything up
    cmip6_dir_fp = Path(
        "/maps/rt582/coralshift/data/env_vars/cmip6/EC-Earth3P-HR/r1i1p2f1/testing/regridded"
    )
    # TODO: remove newtest
    # TODO: include levs check
    correct_area_fps = list(
        functions_creche.find_files_for_area(
            cmip6_dir_fp.rglob("*.nc"), lat_range=lats, lon_range=lons
        ),
    )
    correct_fps = functions_creche.find_files_for_time(
        cmip6_dir_fp.rglob("*.nc"), year_range=sorted(year_range)
    )
    # TODO: check that also spans full year range
    if len(correct_fps) > 0:
        # check that file includes all variables in variables list
        for fp in correct_area_fps:
            if all(variable in str(fp) for variable in variables):
                return (
                    functions_creche.process_xa_d(xa.open_dataset(fp)).sel(
                        latitude=slice(min(lats), max(lats)),
                        longitude=slice(min(lons), max(lons)),
                    ),
                    fp,
                )
    return None, None


def construct_cmip_commands(
    variables,
    source_id: str = "EC-Earth3P-HR",
    member_id: str = "r1i1p2f1",
    lats: list[float, float] = [-40, 0],
    lons: list[float, float] = [130, 170],
    levs: list[int, int] = [0, 20],
    logging_dir: str = config.logs_dir,
) -> (list, list):
    cmip_commands = []
    output_log_paths = []
    if not Path(logging_dir).exists():
        Path(logging_dir).mkdir(parents=True, exist_ok=True)

    for variable in variables:
        # if not, run necessary downloading
        cmip_command = construct_cmip_command(
            variable, source_id, member_id, lats, lons, levs
        )
        cmip_commands.append(cmip_command)

        output_log_paths.append(
            Path(logging_dir) / f"{source_id}_{member_id}_{variable}.txt"
        )
    return cmip_commands, output_log_paths


def generate_remapping_file(
    eg_xa: xa.Dataset | xa.DataArray,
    remap_template_fp: str | Path,
    resolution: float = 0.25,
    out_grid: str = "lonlat",
):
    xsize, ysize, xfirst, yfirst, x_inc, y_inc = generate_remap_info(
        eg_xa, resolution, out_grid
    )

    print(f"Saving regridding info to {remap_template_fp}...")
    with open(remap_template_fp, "w") as file:
        file.write(
            f"gridtype = {out_grid}\n"
            f"xsize = {xsize}\n"
            f"ysize = {ysize}\n"
            f"xfirst = {xfirst}\n"
            f"yfirst = {yfirst}\n"
            f"xinc = {x_inc}\n"
            f"yinc = {y_inc}\n"
        )


def generate_remap_info(
    eg_nc, source_id: str, resolution=0.25, out_grid: str = "lonlat"
):
    # [-180, 180] longitudinal range
    xfirst = float(np.min(eg_nc.longitude).values) - 180
    yfirst = float(np.min(eg_nc.latitude).values)

    xsize = int(360 / resolution)
    # [smallest latitude, largest latitude] range
    ysize = int((180 / resolution) + yfirst)

    x_inc, y_inc = resolution, resolution

    return xsize, ysize, xfirst, yfirst, x_inc, y_inc


def generate_remapping_file(
    eg_xa: xa.Dataset | xa.DataArray,
    remap_template_fp: str | Path,
    resolution: float = 0.25,
    out_grid: str = "lonlat",
):
    xsize, ysize, xfirst, yfirst, x_inc, y_inc = generate_remap_info(
        eg_xa, resolution, out_grid
    )

    print(f"Saving regridding info to {remap_template_fp}...")
    with open(remap_template_fp, "w") as file:
        file.write(
            f"gridtype = {out_grid}\n"
            f"xsize = {xsize}\n"
            f"ysize = {ysize}\n"
            f"xfirst = {xfirst}\n"
            f"yfirst = {yfirst}\n"
            f"xinc = {x_inc}\n"
            f"yinc = {y_inc}\n"
        )


def gen_seafloor_indices(xa_da: xa.Dataset, var: str, dim: str = "lev"):
    """Generate indices of seafloor values for a given variable in an xarray dataset.

    Args:
        xa_da (xa.Dataset): xarray dataset containing variable of interest
        var (str): name of variable of interest
        dim (str, optional): dimension along which to search for seafloor values. Defaults to "lev".

    Returns:
        indices_array (np.ndarray): array of indices of seafloor values for given variable
    """
    nans = np.isnan(xa_da[var]).sum(dim=dim)  # separate out
    indices_array = -(nans.values) - 1
    indices_array[indices_array == -(len(xa_da[dim].values) + 1)] = -1
    return indices_array


def extract_seafloor_vals(xa_da, indices_array):
    vals_array = xa_da.values
    t, j, i = indices_array.shape
    # create open grid for indices along each dimension
    t_grid, j_grid, i_grid = np.ogrid[:t, :j, :i]
    # select values from vals_array using indices_array
    return vals_array[t_grid, indices_array, j_grid, i_grid]


def find_files_for_time(filepaths, year_range):
    result = []
    for fp in filepaths:
        year = int(str(fp).split("_")[-1].split("-")[0][:4])
        if year_range[0] <= year <= year_range[1] - 1:
            result.append(fp)
    return result


def find_files_for_area(filepaths, lat_range, lon_range):
    result = []

    for filepath in filepaths:
        # uncropped refers to global coverage
        if "uncropped" in str(filepath):
            result.append(filepath)
            continue

        fp_lats, fp_lons = extract_lat_lon_ranges_from_fp(filepath)
        if (
            max(lat_range) <= max(fp_lats)
            and min(lat_range) >= min(fp_lats)
            and max(lon_range) <= max(fp_lons)
            and min(lon_range) >= min(fp_lons)
        ):
            result.append(filepath)

    return result


# END OF CMIPPER FILES


# from baselines.py
###############################################


def process_df_for_rfr(
    df: pd.DataFrame,
    predictors: list[str],
    gt: str,
    split_type: str = "spatial",
    train_val_test_frac: list[float] = [0.6, 0.2, 0.2],
) -> np.ndarray:
    """
    Process a dataframe for use with a random forest regressor.

    Args:
        df (pd.DataFrame): The dataframe to process.
        predictors (list[str]): The names of the predictor variables.
        gt (str): The name of the ground truth variable.
        split_type (str, optional): The type of split to perform. Defaults to "spatial".
        train_val_test_frac (list[float], optional): The fractions of the dataframe to allocate to train, test and
        validation sets. Defaults to [0.6, 0.2, 0.2].

    Returns:
        np.ndarray: The predictor and ground truth arrays.

    TODO: tidy optional val handling: is it even necessary to ever specify, since will be using these only (?) with
    sklearn and its CV functionality?
    """
    # split data into train, test, val dfs
    if split_type == "spatial":
        df_ttv_list = ttv_spatial_split(df, train_val_test_frac)
    elif split_type == "pixel":
        print("To implement")

    # generate and save scaling parameters, ignoring nans. Start with min-max scaling
    scaling_params_dict = scaling_params_from_df(df_ttv_list[0], predictors + [gt])
    # scale train, val, test according to train scaling parameters
    df_tvt_scaled_list = [
        scale_dataframe(df, scaling_params_dict) for df in df_ttv_list
    ]
    df_ttv_list = None  # trying to free up memory
    # one-hot encode nans
    df_tvt_scaled_onehot_list = [
        onehot_df(scaled_df) for scaled_df in df_tvt_scaled_list
    ]
    df_tvt_scaled_list = None  # trying to free up memory

    #     def process_df(df, predictors, gt):
    #         if len(df) > 0:
    #             return Xs_ys_from_df(df, predictors + ["onehot_nan"], gt)
    #         else:
    #             return (None, None)

    #     delayed_samples = [delayed(process_df)(ddf, predictors, gt) for ddf in ddf_list]

    #     return delayed_samples, df_tvt_scaled_onehot_list
    samples = []
    for df in df_tvt_scaled_onehot_list:
        if len(df) > 0:
            samples.append(Xs_ys_from_df(df, predictors + ["onehot_nan"], gt))
        else:
            samples.append((None, None))

    return samples, df_tvt_scaled_onehot_list
    # if 0 in train_val_test_frac:
    #     # cast to numpy arrays
    #     trains, vals, tests = [
    #         Xs_ys_from_df(df, predictors + ["onehot_nan"], gt)
    #         for df in df_tvt_scaled_onehot_list if len(df) > 0 else return (0,0)
    #     ]
    #     return (trains, (0, 0), tests), df_tvt_scaled_list
    # else:
    #     # cast to numpy arrays
    #     trains, vals, tests = [
    #         Xs_ys_from_df(df, predictors + ["onehot_nan"], gt)
    #         for df in df_tvt_scaled_onehot_list
    #     ]
    #     return (trains, vals, tests), df_tvt_scaled_list


# functions_creche.py
###############################################


# my files aren't named like this any more
def extract_lat_lon_ranges_from_fp(file_path):
    # Define the regular expression pattern
    pattern = re.compile(
        r".*_n(?P<north>[\d.-]+)_s(?P<south>[\d.-]+)_w(?P<west>[\d.-]+)_e(?P<east>[\d.-]+).*.nc",
        re.IGNORECASE,
    )

    # Match the pattern in the filename
    match = pattern.match(str(Path(file_path.name)))

    if match:
        # Extract latitude and longitude values
        north = float(match.group("north"))
        south = float(match.group("south"))
        west = float(match.group("west"))
        east = float(match.group("east"))

        # Create lists of latitudes and longitudes
        lats = [north, south]
        lons = [west, east]
        return lats, lons
    else:
        return [-9999, -9999], [-9999, -9999]


def xgb_random_search(
    custom_scale_pos_weight: float = None,
    booster: list[str] = ["gbtree", "gblinear", "dart"],
    eta: list[float] = [0.01, 0.1, 0.3, 0.5],
    gamma: list[float] = [0, 0.1, 0.3, 0.5],
    max_depth: list[int] = [3, 5, 7, 10, 50, 100],
    min_child_weight: list[int] = [1, 3, 5, 7],
    max_delta_step: list[float] = [0, 0.1, 0.3, 0.5],
    subsample: list[float] = [0.3, 0.5, 0.7, 0.9],
    sampling_method: list[str] = ["uniform"],
    lambdas: list[float] = [0, 0.1, 0.3, 0.5, 1, 2],
    scale_pos_weight: list[float] = [0.2, 0.4, 0.6, 0.8],
    refresh_leaf: list[int] = [0, 1],
    grow_policy: list[str] = ["depthwise", "lossguide"],
    max_leaves: list[int] = [0, 10, 20, 50, 100],
    max_bin: list[int] = [256, 512, 1024],
) -> dict:
    """
    Takes multiple values for XGBoost hyperparameters and returns a dictionary of parameter: value(s) pairs

    N.B. all can be left as default, but recommend that custom_scale_pos_weight is provided
    """
    # if providing a specific value to scale_pos_weight (e.g. (len(y_train)-sum(y_train))/sum(y_train)),
    # append this value to list
    if custom_scale_pos_weight:
        scale_pos_weight.append(custom_scale_pos_weight)

    param_space = {
        "booster": booster,
        "eta": eta,
        "gamma": gamma,
        "max_depth": max_depth,
        "min_child_weight": min_child_weight,
        "max_delta_step": max_delta_step,
        "subsample": subsample,
        "sampling_method": sampling_method,
        # colsample
        "lambda": lambdas,
        # "treemethod": ["hist", "approx"],    # default is auto, which is same as hist. One of these two required for growpolicy. BUt not used # noqa
        "scale_pos_weight": scale_pos_weight,
        "refresh_leaf": refresh_leaf,
        # process_type
        "grow_policy": grow_policy,
        "max_leaves": max_leaves,
        "max_bin": max_bin,
    }
    return param_space


def pd_to_array(df):
    """Converts pandas dataframe to xarray instance"""
    return df.to_xarray().sortby("longitude").sortby("latitude")


# no longer returning as numpy: pandas keeps more data
def Xs_ys_from_df(
    df: pd.DataFrame, predictors: list[str], gt: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Splits a dataframe into predictors and ground truth arrays.

    Args:
        df (pd.DataFrame): The dataframe to split.
        predictors (list[str]): The names of the predictor variables.
        gt (str): The name of the ground truth variable.

    Returns:
        tuple[np.ndarray, np.ndarray]: The predictor and ground truth arrays.
    """
    # split df into predictors and gt
    return df[predictors].to_numpy(), df[gt].to_numpy()


# currently handled by spatial_predictions_from_data in spatial_data
def reform_df(df: pd.DataFrame | pd.Series, predictions: np.ndarray) -> pd.DataFrame:
    """
    Reformats a DataFrame to include a column of predictions.
    """
    assert len(df) == len(
        predictions
    ), "Length of DataFrame and predictions array must be equal."

    # Create a copy of the original DataFrame to avoid modifying it in place
    df_predictions = df.copy()

    if isinstance(predictions, np.ndarray):
        predictions_series = pd.Series(
            predictions, index=df_predictions.index, name="predictions"
        )

    # add prediction column to df_predictions
    return pd.concat([df_predictions, predictions_series], axis=1)


# from xgb_funcs.py
###############################################


# doing too much, been adapted since in static_models
# def sklearn_random_search(
#     X, y, n_jobs: int = int(multiprocessing.cpu_count() * FRAC_COMPUTE)
# ):
#     """
#     Perform a random search for the best hyperparameters for an XGBoost model.
#     Must only fit on training set, to avoid leakage onto test/validation set.
#     """
#     print("Parallel Parameter optimization...")
#     # Make sure the number of threads is balanced.
#     xgb_model = xgb.XGBRegressor(n_jobs=n_jobs, tree_method="hist")

#     param_space = functions_creche.xgb_random_search(custom_scale_pos_weight=None)

#     # time execution
#     start_time = time.time()

#     reg = RandomizedSearchCV(
#         estimator=xgb_model,
#         param_distributions=param_space,
#         cv=CV_FOLDS,
#         n_iter=N_ITER,
#         scoring="neg_root_mean_squared_error",  # TODO: look int other metrics
#         verbose=1,
#         n_jobs=n_jobs,
#     )
#     reg.fit(X, y)
#     end_time = time.time()

#     # Calculate and print the elapsed time
#     elapsed_time = end_time - start_time
#     print(f"Elapsed Time: {elapsed_time:.02f} seconds")

#     print(f"Best score: {reg.best_score_:.02f}")
#     print("Best parameters: ", reg.best_params_)


# too specific, not stringable together. Unnecessary random assignment: should just use defaults
# def xgb_baseline(dtrain):
#     param_space = functions_creche.xgb_random_search(custom_scale_pos_weight=None)
#     # randomly choose a value for each hyperparameter
#     params = {key: np.random.choice(value) for key, value in param_space.items()}
#     params["eval_metric"] = "rmse"
#     # setting large num_boost_round, for which optimal number is hopefully less
#     num_boost_round = 999

#     # get mean value from training data
#     mean_train = np.mean(y_train)
#     # baseline predictions
#     baseline_preds = np.full_like(y_train, mean_train)
#     # baseline RMSE
#     baseline_rmse = root_mean_squared_error(y_train, baseline_preds)

# # super deprecated XGBoost pipeline, but not awful
# def main():
#     # DATA LOADING
#     # load data. TODO: add multifile loading for machines with reasonable numbers of CPUs
#     print("loading static data...")
#     all_data_df = load_static_data(RES_STR)

#     # DATA PREPROCESSING
#     # apply depth mask
#     print("preprocessing...")
#     depth_masked_df = functions_creche.depth_filter(
#         all_data_df, depth_mask_lims=DEPTH_MASK_LIMS
#     )
#     # split data into train, test, validation
#     [train_df, test_df, val_df] = functions_creche.train_test_val_split(
#         depth_masked_df, ttv_fractions=TTV_FRACTIONS, split_method=SPLIT_METHOD
#     )
#     # preprocess data: initialise scalers values
#     scaler_X = generate_data_scaler(SCALE_METHOD_X)
#     scaler_y = generate_data_scaler(SCALE_METHOD_y)
#     # preprocess data: nan handling
#     [train_df, test_df, val_df] = [
#         onehot_nan(df, REMOVE_NAN_ROWS) for df in [train_df, test_df, val_df]
#     ]
#     # split into X and y for each subset
#     (train_X, train_y), (test_X, test_y), (val_X, val_y) = [
#         split_X_y(df, GROUND_TRUTH_NAME) for df in [train_df, test_df, val_df]
#     ]
#     # fit scalers
#     scaler_X.fit(train_X)
#     scaler_y.fit(train_y)
#     # transform data
#     (train_X, train_y), (test_X, test_y), (val_X, val_y) = [
#         (scaler_X.transform(X), scaler_y.transform(y))
#         for X, y in [(train_X, train_y), (test_X, test_y), (val_X, val_y)]
#     ]
#     # cast data to DMatrix for native xgboost handling
#     dtrain, dtest, dval = [
#         xgb.DMatrix(X, y)
#         for X, y in [(train_X, train_y), (test_X, test_y), (val_X, val_y)]
#     ]

#     # MODEL FITTING
#     print("fitting model...")
#     sklearn_random_search(train_X[:DATA_SUBSET], train_y[:DATA_SUBSET])


# if __name__ == "__main__":
#     main()


# from dataloading/bathymetry.py
###############################################


# for 30m GBR data. Could be useful in future, but probably ways to improve
class ReefAreas:
    name_mapping = {
        "A": "Great_Barrier_Reef_A_2020_30m",
        "B": "Great_Barrier_Reef_B_2020_30m",
        "C": "Great_Barrier_Reef_C_2020_30m",
        "D": "Great_Barrier_Reef_D_2020_30m",
        # Add more mappings as needed
    }

    def __init__(self):
        self.datasets = [
            {
                "name": "Great_Barrier_Reef_A_2020_30m",
                "short_name": "A",
                "file_name": "Great_Barrier_Reef_A_2020_30m_MSL_cog.tif",
                "short_file_name": "Great_Barrier_Reef_A",
                "xarray_name": "bathymetry_A",
                "lat_range": (-10, -17),
                "lon_range": (142, 147),
                "url": "https://ausseabed-public-warehouse-bathymetry.s3.ap-southeast-2.amazonaws.com/L3/0b9ad3f3-7ade-40a7-ae70-f7c6c0f3ae2e/Great_Barrier_Reef_A_2020_30m_MSL_cog.tif",  # noqa
            },
            {
                "name": "Great_Barrier_Reef B 2020 30m",
                "short_name": "B",
                "file_name": "Great_Barrier_Reef_B_2020_30m_MSL_cog.tif",
                "short_file_name": "Great_Barrier_Reef_B",
                "xarray_name": "bathymetry_B",
                "lat_range": (-16, -23),
                "lon_range": (144, 149),
                "url": "https://ausseabed-public-warehouse-bathymetry.s3.ap-southeast-2.amazonaws.com/L3/4a6e7365-d7b1-45f9-a576-2be8ff8cd755/Great_Barrier_Reef_B_2020_30m_MSL_cog.tif",  # noqa
            },
            {
                "name": "Great_Barrier_Reef C 2020 30m",
                "short_name": "C",
                "file_name": "Great_Barrier_Reef_C_2020_30m_MSL_cog.tif",
                "short_file_name": "Great_Barrier_Reef_C",
                "xarray_name": "bathymetry_C",
                "lat_range": (-18, -24),
                "lon_range": (148, 154),
                "url": "https://ausseabed-public-warehouse-bathymetry.s3.ap-southeast-2.amazonaws.com/L3/3b171f8d-9248-4aeb-8b32-0737babba3c2/Great_Barrier_Reef_C_2020_30m_MSL_cog.tif",  # noqa
            },
            {
                "name": "Great_Barrier_Reef D 2020 30m",
                "short_name": "D",
                "file_name": "Great_Barrier_Reef_D_2020_30m_MSL_cog.tif",
                "short_file_name": "Great_Barrier_Reef_D",
                "xarray_name": "bathymetry_D",
                "lat_range": (-23, -29),
                "lon_range": (150, 156),
                "url": "https://ausseabed-public-warehouse-bathymetry.s3.ap-southeast-2.amazonaws.com/L3/7168f130-f903-4f2b-948b-78508aad8020/Great_Barrier_Reef_D_2020_30m_MSL_cog.tif",  # noqa
            },
        ]

    def get_long_name_from_short(self, short_name):
        return self.name_mapping.get(short_name, "Unknown")

    def get_name_from_names(self, name):
        for dataset in self.datasets:
            if (
                dataset["name"] == name
                or dataset["short_name"] == name
                or dataset["file_name"] == name
                or dataset["short_file_name"] == name
            ):
                return dataset["name"]
        raise ValueError(f"'{name}' not a dataset.")

    def get_filename(self, name):
        name = self.get_name_from_names(name)
        dataset = self.get_dataset(name)
        if dataset:
            return dataset["file_name"]
        return None

    def get_short_filename(self, name):
        name = self.get_name_from_names(name)
        dataset = self.get_dataset(name)
        if dataset:
            return dataset["short_file_name"]
        return None

    def get_letter(self, name):
        name = self.get_name_from_names(name)
        dataset = self.get_dataset(name)
        if dataset:
            return dataset["short_name"]
        return None

    def get_xarray_name(self, name):
        name = self.get_name_from_names(name)
        dataset = self.get_dataset(name)
        if dataset:
            return dataset["xarray_name"]
        return None

    def get_dataset(self, name):
        name = self.get_name_from_names(name)
        for dataset in self.datasets:
            if dataset["name"] == name:
                return dataset
        return None

    def get_lat_lon_limits(self, name):
        dataset = self.get_dataset(name)
        if dataset:
            return dataset["lat_range"], dataset["lon_range"]
        return None

    def get_url(self, name):
        name = self.get_name_from_names(name)
        dataset = self.get_dataset(name)
        if dataset:
            return dataset["url"]
        return None


def ensure_bathymetry_downloaded(area_name: str, loading_bar: bool = True) -> ReefAreas:
    """
    Ensures that the bathymetry data for the specified area is downloaded.

    Parameters
    ----------
        area_name (str): The name of the area.
        loading_bar (bool, optional): Whether to display a loading bar during the download process.
                                      Defaults to True.

    Returns
    -------
        ReefAreas: An instance of the ReefAreas class representing information about the downloaded area, and the other
        potentials.
    """
    areas_info = ReefAreas()

    # get url
    area_url = areas_info.get_url(area_name)
    # generate path to save data to
    save_path = directories.get_bathymetry_datasets_dir() / areas_info.get_filename(
        area_name
    )
    # download data if not already there
    file_ops.check_exists_download_url(save_path, area_url)

    return ReefAreas()


######################################################
# Not used within MRes: coastal bathymetry for US only
######################################################

# these two imports necessary for importing US coastal bathymetry data
# import requests
# from bs4 import BeautifulSoup

# def fetch_links_from_url(page_url: str, suffix: str = None) -> list[str]:
#     """Fetches all links (as href attributes) from a webpage and returns a list of them. If a `suffix` argument is
#     provided, only the links that end with that suffix are included.

#     Parameters
#     ----------
#         page_url (str): The URL of the webpage to fetch links from.
#         suffix (str, optional): The suffix to filter links by. Defaults to None.

#     Returns
#     -------
#         list[str]: A list of links from the webpage.
#     """
#     reqs = requests.get(page_url)
#     soup = BeautifulSoup(reqs.text, "html.parser")
#     link_segments = soup.find_all("a")
#     # extract link strings, excluding None values
#     links = [link.get("href") for link in link_segments if link.get("href") is not None]

#     if suffix:
#         links = [link for link in links if link.endswith(file_ops.pad_suffix(suffix))]

#     return links


# def download_etopo_data(
#     download_dest_dir: Path | str,
#     resolution: str | int = 15,
#     file_extension: str = ".nc",
#     loading_bar: bool = True,
# ) -> None:
#     """
#     Downloads ETOPO data files from the NOAA website and saves them to the specified directory.

#     Parameters
#     ----------
#         download_dest_dir (Path | str): The directory to save the downloaded files to.
#         resolution (str | int, optional): The resolution of the data in degrees (15, 30, 60). Defaults to 15.
#         file_extension (str, optional): The file extension to filter downloads by. Defaults to '.nc'.
#         loading_bar (bool, optional): Whether to display a progress bar during download. Defaults to True.

#     Returns
#     -------
#         None
#     """
#     file_server_url = "https://www.ngdc.noaa.gov/thredds/fileServer/global/"
#     page_url = f"https://www.ngdc.noaa.gov/thredds/catalog/global/ETOPO2022/{resolution}s/{resolution}s_geoid_netcdf/catalog.html"  # noqa

#     for link in fetch_links_from_url(page_url, file_extension):
#         # file url involves multiple levels
#         file_specifier = file_ops.get_n_last_subparts_path(Path(link), 4)
#         file_name = file_ops.get_n_last_subparts_path(Path(link), 1)

#         url = file_server_url + str(file_specifier)
#         download_dest_path = Path(download_dest_dir, file_name)

#         file_ops.check_exists_download_url(download_dest_path, url, loading_bar)


def generate_gradient_magnitude_ncs(
    regions: list[str] = ["A", "B", "C", "D"],
    resolution_d: float = None,
    sigma: int = 1,
):
    bath_dir = directories.get_bathymetry_datasets_dir()

    res_string = utils.generate_resolution_str(resolution_d=resolution_d)
    for region in tqdm(
        regions,
        total=len(regions),
        desc=f" Generating seafloor slopes nc files at {resolution_d:.04f}",
    ):
        region_name = ReefAreas().get_short_filename(region)
        bath_file = list(bath_dir.glob(f"{region_name}_*{res_string[:3]}*.nc"))[0]
        bath_da = file_ops.open_xa_file(bath_file)
        _, _ = generate_gradient_magnitude_nc(bath_da, resolution_d=resolution_d)


# should split up generating and saving
def generate_gradient_magnitude_nc(
    xa_da: xa.DataArray, resolution_d: float = None, sigma: int = 1
):
    """
    Generate a NetCDF file containing the gradient magnitude of a DataArray. The NetCDF file is saved in the directory
    specified by directories.get_gradients_dir(), with the filename format
    "{data_array_name}_{spatial_resolution}_gradients.nc".

    Parameters
    ----------
        xa_da (xarray.DataArray): The input DataArray.
        sigma (int, optional): Standard deviation of the Gaussian filter. Default is 1.

    Returns
    -------
        tuple[xa.DataArray, str]: A tuple containing the gradient DataArray and the path to the saved NetCDF file.
    """
    # generate savepath
    if not resolution_d:
        resolution_d = np.mean(spatial_data.calculate_spatial_resolution(xa_da))
    res_string = utils.generate_resolution_str(resolution_d)
    filename = f"{xa_da.name}_{res_string}_gradients.nc"
    save_path = directories.get_gradients_dir() / filename

    # generate/save file as necessary
    if not save_path.is_file():
        grad_da = calculate_gradient_magnitude(xa_da, sigma)
        grad_da.to_netcdf(save_path)
        print(f"{filename} saved at {save_path}.")
    else:
        print(f"{filename} already exists at {save_path}.")
        grad_da = xa.open_dataarray(save_path)

    return grad_da, save_path


def generate_bathymetry_xa_da(area_name: str):
    """
    Generate bathymetry data for a specified area.

    Parameters
    ----------
        area_name (str): The name of the area.

    Returns
    -------
        tuple[str, xa.DataArray]: A tuple containing the filepath and the processed xarray of the generated bathymetry
        data.
    """
    # download .tif if not downloaded aready
    reef_areas = ensure_bathymetry_downloaded(area_name)
    # cast tif to processed xarray with correct crs
    xa_bath = spatial_data.tif_to_xarray(
        directories.get_bathymetry_datasets_dir() / reef_areas.get_filename(area_name),
        reef_areas.get_xarray_name(area_name),
    )

    xa_bath.rio.write_crs("EPSG:4326", inplace=True)

    resolution = np.mean(spatial_data.calculate_spatial_resolution(xa_bath))

    bath_name = f"{reef_areas.get_xarray_name(area_name)}_{resolution:.05f}d"
    filepath, xa_da = file_ops.save_nc(
        directories.get_bathymetry_datasets_dir(), bath_name, xa_bath, return_array=True
    )

    return filepath, xa_da


# from dataloading/climate_data.py
###############################################

# some decent functions for automating download of ERA5 and Global reanalysis datasets.
# May be useful in future

import numpy as np
import xarray as xa
import os
import cdsapi
import getpass

from pathlib import Path
from tqdm import tqdm
from pandas._libs.tslibs.timestamps import Timestamp

from coralshift.utils import utils, file_ops, directories
from coralshift.processing import spatial_data
from coralshift.dataloading import bathymetry


def generate_spatiotemporal_var_filename_from_dict(
    info_dict: dict,
) -> str:
    """Generate a filename based on variable, date, and coordinate limits.

    Parameters
    ----------
    info_dict (dict): A dictionary containing information about the variable, date, and coordinate limits.

    Returns
    -------
    str: The generated filename.
    """
    filename_list = []
    for k, v in info_dict.items():
        # strings (variables)
        if utils.is_type_or_list_of_type(v, str):
            filename_list.extend(
                [
                    k.upper(),
                    utils.replace_dot_with_dash(utils.underscore_str_of_strings(v)),
                ]
            )
        # np.datetime64 (dates)
        elif utils.is_type_or_list_of_type(
            v, np.datetime64
        ) or utils.is_type_or_list_of_type(v, Timestamp):
            filename_list.extend(
                [
                    k.upper(),
                    utils.replace_dot_with_dash(utils.underscore_str_of_dates(v)),
                ]
            )
        # tuples (coordinates limits)
        elif utils.is_type_or_list_of_type(v, tuple):
            filename_list.extend(
                [
                    k.upper(),
                    utils.replace_dot_with_dash(
                        utils.underscore_list_of_tuples(utils.round_list_tuples(v))
                    ),
                ]
            )
    return "_".join(filename_list)


def generate_metadata(
    download_dir: str,
    filename: str,
    variables: list[str],
    date_lims: tuple[str, str],
    lon_lims: list[float],
    lat_lims: list[float],
    depth_lims: list[float],
    query: str = "n/a",
) -> None:
    """Generate metadata for the downloaded file and save it as a JSON file.

    Parameters
    ----------
    download_dir (str): The directory where the file is downloaded.
    filename (str): The name of the file.
    variable (str): The variable acronym.
    date_lims (tuple[str, str]): A tuple containing the start and end dates.
    lon_lims (list[float]): A list containing the longitude limits.
    lat_lims (list[float]): A list containing the latitude limits.
    depth_lims (list[float]): A list containing the depth limits.
    query (str): The MOTU query used for downloading the file.
    """
    filepath = (Path(download_dir) / filename).with_suffix(".json")

    var_dict = {
        "mlotst": "ocean mixed layer thickness (sigma theta)",
        "siconc": "sea ice area fraction",
        "thetao": "sea water potential temperature",
        "usi": "eastward sea ice velocity",
        "sithick": "sea ice thickness",
        "bottomT": "sea water potential temperature at sea floor",
        "vsi": "northward sea ice velocity",
        "usi": "eastward sea ice velocity",
        "vo": "northward sea water velocity",
        "uo": "eastward sea water velocity",
        "so": "sea water salinity",
        "zos": "sea surface height above geoid",
    }

    # if list of variables, iterate through dict to get long names
    if len(variables) > 1:
        variable_names = str([var_dict[var] for var in variables])
    else:
        variable_names = var_dict[variables[0]]
    # send list to a string for json
    variables = str(variables)

    metadata = {
        "filename": filename,
        "download directory": str(download_dir),
        "variable acronym(s)": variables,
        "variable name(s)": variable_names,
        "longitude-min": lon_lims[0],
        "longitude-max": lon_lims[1],
        "latitude-min": lat_lims[0],
        "latitude-max": lat_lims[1],
        "date-min": str(date_lims[0]),
        "date-max": str(date_lims[1]),
        "depth-min": depth_lims[0],
        "depth-max": depth_lims[1],
        "motu_query": query,
    }
    # save metadata as json file at filepath
    file_ops.save_json(metadata, filepath)


def generate_name_dict(
    variables: list[str],
    date_lims: tuple[str, str],
    lon_lims: tuple[str, str],
    lat_lims: tuple[str, str],
    depth_lims: tuple[str, str],
) -> dict:
    return {
        "vars": variables,
        "dates": date_lims,
        "lons": lon_lims,
        "lats": lat_lims,
        "depths": depth_lims,
    }


def download_reanalysis(
    download_dir: str | Path,
    region: str,
    final_filename: str = None,
    variables: list[str] = ["mlotst", "bottomT", "uo", "so", "zos", "thetao", "vo"],
    date_lims: tuple[str, str] = ("1992-12-31", "2020-12-16"),
    depth_lims: tuple[str, str] = (0.3, 1),
    lon_lims: tuple[str, str] = (142, 147),
    lat_lims: tuple[str, str] = (-17, -10),
    product_type: str = "my",
    service_id: str = "GLOBAL_MULTIYEAR_PHY_001_030",
    product_id: str = "cmems_mod_glo_phy_my_0.083_P1D-m",
) -> xa.Dataset:
    """
    Download reanalysis data for multiple variables and save them to the specified directory.

    Parameters
    ----------
    download_dir (str | Path): Directory to save the downloaded files.
    variables (list[str]): List of variables to download.
    date_lims (tuple[str, str]): Date limits as a tuple of strings in the format "YYYY-MM-DD".
    lon_lims (tuple[str, str]): Longitude limits as a tuple of strings in the format "lon_min, lon_max".
    lat_lims (tuple[str, str]): Latitude limits as a tuple of strings in the format "lat_min, lat_max".
    depth_lims (tuple[str, str]): Depth limits as a tuple of strings in the format "depth_min, depth_max".
    product_type (str, optional): Product type. Defaults to "my".
    service_id (str, optional): Product ID. Defaults to "GLOBAL_MULTIYEAR_PHY_001_030".
    product_id (str, optional): Dataset ID. Defaults to "cmems_mod_glo_phy_my_0.083_P1D-m".

    Returns
    -------
    xa.Dataset: dataset merged from individual files

    Notes
    -----
    Currently taking only topmost depth (TODO: make full use of profile)

    """
    download_dir = file_ops.guarantee_existence(Path(download_dir) / region)
    merged_download_dir = file_ops.guarantee_existence(download_dir / "merged_vars")

    # User credentials
    username = input("Enter your username: ")
    password = getpass.getpass("Enter your password: ")

    # generate name of combined file
    name_dict = generate_name_dict(variables, date_lims, lon_lims, lat_lims, depth_lims)
    main_filename = generate_spatiotemporal_var_filename_from_dict(name_dict)
    save_path = (Path(download_dir) / main_filename).with_suffix(".nc")

    # if particular filename specified
    if final_filename:
        save_path = (
            Path(download_dir) / file_ops.remove_suffix(final_filename)
        ).with_suffix(".nc")

    if save_path.is_file():
        print(f"Merged file already exists at {save_path}")
        return xa.open_dataset(save_path), save_path

    date_merged_xas = []
    # split request by variable
    for var in tqdm(variables, desc=" variable loop", position=0, leave=True):
        print(f"Downloading {var} data...")
        # split request by time
        date_pairs = utils.generate_date_pairs(date_lims)
        # create download folder for each variable (if not already existing)
        save_dir = Path(download_dir) / var
        file_ops.guarantee_existence(save_dir)
        for sub_date_lims in tqdm(date_pairs, leave=False):
            # generate name info dictionary
            name_dict = generate_name_dict(
                var, sub_date_lims, lon_lims, lat_lims, depth_lims
            )
            filename = generate_spatiotemporal_var_filename_from_dict(name_dict)
            # if file doesn't already exist, generate and execute API query
            # print((Path(save_dir) / filename).with_suffix(".nc"))
            if not (Path(save_dir) / filename).with_suffix(".nc").is_file():
                query = generate_motu_query(
                    save_dir,
                    filename,
                    var,
                    sub_date_lims,
                    lon_lims,
                    lat_lims,
                    depth_lims,
                    product_type,
                    service_id,
                    product_id,
                    username,
                    password,
                )
                execute_motu_query(
                    save_dir,
                    filename,
                    [var],
                    sub_date_lims,
                    lon_lims,
                    lat_lims,
                    depth_lims,
                    query,
                )
            else:
                print(f"{filename} already exists in {save_dir}.")

        var_name_dict = generate_name_dict(
            var, date_lims, lon_lims, lat_lims, depth_lims
        )
        date_merged_name = generate_spatiotemporal_var_filename_from_dict(var_name_dict)
        # merge files by time
        merged_path = file_ops.merge_nc_files_in_dir(
            save_dir,
            date_merged_name,
            merged_save_path=(merged_download_dir / date_merged_name).with_suffix(
                ".nc"
            ),
        )

        # generate metadata if necessary
        if not (merged_download_dir / date_merged_name).with_suffix(".json").is_file():
            generate_metadata(
                merged_download_dir,
                date_merged_name,
                [var],
                date_lims,
                lon_lims,
                lat_lims,
                depth_lims,
            )

        date_merged_xas.append(merged_path)

    # concatenate variables
    arrays = [
        (
            spatial_data.process_xa_d(
                xa.open_dataset(date_merged_xa, decode_coords="all")
            )
        )
        for date_merged_xa in date_merged_xas
    ]
    all_merged = xa.merge(arrays)

    save_nc(download_dir, final_filename, all_merged)
    # all_merged.to_netcdf(save_path)
    # generate accompanying metadata
    generate_metadata(
        download_dir,
        final_filename,
        variables,
        date_lims,
        lon_lims,
        lat_lims,
        depth_lims,
    )
    return all_merged, save_path


def save_nc(
    save_dir: Path | str,
    filename: str,
    xa_d: xa.DataArray | xa.Dataset,
    return_array: bool = False,
) -> xa.DataArray | xa.Dataset:
    """
    Save the given xarray DataArray or Dataset to a NetCDF file iff no file with the same
    name already exists in the directory.
    # TODO: issues when suffix provided
    Parameters
    ----------
        save_dir (Path or str): The directory path to save the NetCDF file.
        filename (str): The name of the NetCDF file.
        xa_d (xarray.DataArray or xarray.Dataset): The xarray DataArray or Dataset to be saved.

    Returns
    -------
        xarray.DataArray or xarray.Dataset: The input xarray object.
    """
    filename = file_ops.remove_suffix(utils.replace_dot_with_dash(filename))
    save_path = (Path(save_dir) / filename).with_suffix(".nc")
    if not save_path.is_file():
        if "grid_mapping" in xa_d.attrs:
            del xa_d.attrs["grid_mapping"]
        print(f"Writing {filename} to file at {save_path}")
        spatial_data.process_xa_d(xa_d).to_netcdf(save_path)
        print("Writing complete.")
    else:
        print(f"{filename} already exists in {save_dir}")

    if return_array:
        return save_path, xa.open_dataset(save_path)
    else:
        return save_path


def execute_motu_query(
    download_dir: str | Path,
    filename: str,
    var: list[str] | str,
    sub_date_lims: tuple[np.datetime64, np.datetime64],
    lon_lims: tuple[float, float],
    lat_lims: tuple[float, float],
    depth_lims: tuple[float, float],
    query: str,
) -> None:
    """Execute the MOTU query to download the data slice and generate metadata.

    Parameters
    ----------
    download_dir (str | Path): The directory where the file will be downloaded.
    filename (str): The name of the file.
    var (list[str] | str): The variable(s) to be downloaded.
    sub_date_lims (tuple[np.datetime64, np.datetime64]): A tuple containing the start and end dates.
    lon_lims (tuple[float, float]): A tuple containing the longitude limits.
    lat_lims (tuple[float, float]): A tuple containing the latitude limits.
    depth_lims (tuple[float, float]): A tuple containing the depth limits.
    query (str): The MOTU query used for downloading the file.

    Returns
    -------
    None
    """
    outcome = download_data_slice(query)
    # if download successful
    if outcome:
        generate_metadata(
            download_dir,
            filename,
            var,
            sub_date_lims,
            lon_lims,
            lat_lims,
            depth_lims,
            query,
        )
        print(f"{filename} written to {download_dir} and metadata generated.")


def generate_motu_query(
    download_dir: str | Path,
    filename: str,
    variable: list[str] | str,
    date_lims: tuple[np.datetime64, np.datetime64],
    lon_lims: tuple[float, float],
    lat_lims: tuple[float, float],
    depth_lims: tuple[float, float],
    product_type: str,
    service_id: str,
    product_id: str,
    username: str,
    password: str,
) -> str:
    """Generate the MOTU query for downloading climate data.

    Parameters
    ----------
    download_dir (str | Path): The directory where the file will be downloaded.
    filename (str): The name of the file.
    variable (list[str] | str): The variable(s) to be downloaded.
    date_lims (tuple[np.datetime64, np.datetime64]): A tuple containing the start and end dates.
    lon_lims (tuple[float, float]): A tuple containing the longitude limits.
    lat_lims (tuple[float, float]): A tuple containing the latitude limits.
    depth_lims (tuple[float, float]): A tuple containing the depth limits.
    product_type (str): The type of product.
    service_id (str): The product ID.
    product_id (str): The dataset ID.
    username (str): The username for authentication.
    password (str): The password for authentication.

    Returns
    -------
    query (str): The MOTU query.

    """

    lon_min, lon_max = min(lon_lims), max(lon_lims)
    lat_min, lat_max = min(lat_lims), max(lat_lims)
    date_min, date_max = min(date_lims), max(date_lims)
    depth_min, depth_max = min(depth_lims), max(depth_lims)

    # generate motuclient command line
    # specifying environment to conda environment makes sure that `motuclient` module is found
    query = f"conda run -n coralshift python -m motuclient \
    --motu https://{product_type}.cmems-du.eu/motu-web/Motu --service-id {service_id}-TDS --product-id {product_id} \
    --longitude-min {lon_min} --longitude-max {lon_max} --latitude-min {lat_min} --latitude-max {lat_max} \
    --date-min '{date_min}' --date-max '{date_max}' --depth-min {depth_min} --depth-max {depth_max} \
    --variable {variable} --out-dir '{download_dir}' --out-name '{filename}.nc' --user '{username}' --pwd '{password}'"

    return query


def download_data_slice(query):
    """Download the data slice using the MOTU query.

    Parameters
    ----------
    query (str): The MOTU query.

    Returns
    -------
    bool: True if the download is successful, False otherwise.
    """
    try:
        os.system(query)
        return True
    except ConnectionAbortedError():
        print("Data download failed.")
        return False


def ecmwf_api_call(
    c,
    filepath: str,
    parameter: str,
    time_info_dict: dict,
    area: list[tuple[float]],
    dataset_tag: str = "reanalysis-era5-single-levels",
    format: str = "nc",
):
    api_call_dict = generate_ecmwf_api_dict(parameter, time_info_dict, area, format)
    # make api call
    try:
        c.retrieve(dataset_tag, api_call_dict, filepath)
    # if error in fetching, limit the parameter
    except ConnectionAbortedError():
        print(f"API call failed for {parameter}.")


def generate_ecmwf_api_dict(
    weather_params: list[str], time_info_dict: dict, area: list[float], format: str
) -> dict:
    """Generate api dictionary format for single month of event"""

    # if weather_params

    api_call_dict = {
        "product_type": "reanalysis",
        "variable": [weather_params],
        "area": area,
        "format": format,
    } | time_info_dict

    return api_call_dict


def return_full_ecmwf_weather_param_strings(dict_keys: list[str]):
    """Look up weather parameters in a dictionary so they can be entered as short strings rather than typed out in full.
    Key:value pairs ordered in expected importance

    Parameters
    ----------
    dict_keys : list[str]
        list of shorthand keys for longhand weather parameters. See accompanying documentation on GitHub
    """

    weather_dict = {
        "d2m": "2m_dewpoint_temperature",
        "t2m": "2m_temperature",
        "skt": "skin_temperature",
        "tp": "total_precipitation",
        "sp": "surface_pressure",
        "src": "skin_reservoir_content",
        "swvl1": "volumetric_soil_water_layer_1",
        "swvl2": "volumetric_soil_water_layer_2",
        "swvl3": "volumetric_soil_water_layer_3",
        "swvl4": "volumetric_soil_water_layer_4",
        "slhf": "surface_latent_heat_flux",
        "sshf": "surface_sensible_heat_flux",
        "ssr": "surface_net_solar_radiation",
        "str": "surface_net_thermal_radiation",
        "ssrd": "surface_solar_radiation_downwards",
        "strd": "surface_thermal_radiation_downwards",
        "e": "total_evaporation",
        "pev": "potential_evaporation",
        "ro": "runoff",
        "ssro": "sub-surface_runoff",
        "sro": "surface_runoff",
        "u10": "10m_u_component_of_wind",
        "v10": "10m_v_component_of_wind",
    }

    weather_params = []
    for key in dict_keys:
        weather_params.append(weather_dict.get(key))

    return weather_params


def hourly_means_to_daily(hourly_dir: Path | str, suffix: str = "netcdf"):
    filepaths = file_ops.return_list_filepaths(hourly_dir, suffix, incl_subdirs=True)
    # create subdirectory to store averaged files
    daily_means_dir = file_ops.guarantee_existence(Path(hourly_dir) / "daily_means")
    for filepath in tqdm(filepaths, desc="Converting hourly means to daily means"):
        filename = "_".join((str(filepath.stem), "daily"))
        save_path = (daily_means_dir / filename).with_suffix(
            file_ops.pad_suffix(suffix)
        )
        # open dataset
        hourly = xa.open_dataset(filepath, chunks={"time": 100})
        daily = hourly.resample(time="1D").mean()
        # take average means
        daily.to_netcdf(save_path)


def generate_month_day_hour_list(items_range):
    items = []

    if isinstance(items_range, (int, np.integer)):
        items_range = [items_range]
    elif isinstance(items_range, np.ndarray):
        items_range = items_range.tolist()
    elif not isinstance(items_range, list):
        raise ValueError(
            "Invalid input format. Please provide an integer, a list, or a NumPy array."
        )

    for item in items_range:
        if isinstance(item, (int, np.integer)):
            if item < 0 or item > 31:
                raise ValueError("Invalid items value: {}.".format(item))
            items.append(item)
        else:
            raise ValueError(
                "Invalid input format. Please provide an integer, a list, or a NumPy array."
            )

    return items


def return_times_info(
    year: int,
    months: list[int] | int = np.arange(1, 13),
    days: list[int] | int = np.arange(1, 32),
    hours: list[int] | int = np.arange(0, 24),
):
    year = str(year)
    months = [
        utils.pad_number_with_zeros(month)
        for month in generate_month_day_hour_list(months)
    ]
    days = [
        utils.pad_number_with_zeros(day) for day in generate_month_day_hour_list(days)
    ]

    hours = [
        utils.pad_number_with_zeros(hour)
        for hour in generate_month_day_hour_list(hours)
    ]
    for h, hour in enumerate(hours):
        hours[h] = f"{hour}:00"

    return {"year": year, "month": months, "day": days, "time": hours}


def fetch_weather_data(
    download_dest_dir,
    weather_params,
    years,
    months: list[int] | int = np.arange(1, 13),
    days: list[int] | int = np.arange(1, 32),
    hours: list[int] | int = np.arange(0, 24),
    lat_lims=(-10, -17),
    lon_lims=(142, 147),
    dataset_tag: str = "reanalysis-era5-single-levels",
    format: str = "netcdf",
):
    c = cdsapi.Client()

    area = [max(lat_lims), min(lon_lims), min(lat_lims), max(lon_lims)]

    for param in weather_params:
        param_download_dest = file_ops.guarantee_existence(
            Path(download_dest_dir) / param
        )
        for year in years:
            filename = generate_spatiotemporal_var_filename_from_dict(
                {"var": param, "lats": lat_lims, "lons": lon_lims, "year": str(year)}
            )
            filepath = str(
                file_ops.generate_filepath(param_download_dest, filename, format)
            )

            if not Path(filepath).is_file():
                time_info_dict = return_times_info(year, months, days)
                ecmwf_api_call(
                    c, filepath, param, time_info_dict, area, dataset_tag, format
                )
            else:
                print(f"Filepath already exists: {filepath}")


def generate_era5_data(
    weather_params: list[float] = [
        "evaporation",
        "significant_height_of_combined_wind_waves_and_swell",
        "surface_net_solar_radiation",
        "surface_pressure",
    ],
    region: str = None,
    years: list[int] = np.arange(1993, 2021),
    lat_lims: tuple[float] = (-10, -17),
    lon_lims: tuple = (142, 147),
) -> None:
    """
    Generates and merges ERA5 weather data files.

    Parameters
    ----------
        weather_params (list[str]): A list of weather parameters to download and merge.
        years (list[int]): A list of years for which to download and merge the data.
        lat_lims (tuple[float, float]): A tuple specifying the latitude limits.
        lon_lims (tuple[float, float]): A tuple specifying the longitude limits.

    Returns
    -------
    None
    """
    save_dir = directories.get_era5_data_dir()
    if region:
        save_dir = save_dir / bathymetry.ReefAreas().get_short_filename(region)
        (lat_lims, lon_lims) = bathymetry.ReefAreas().get_lat_lon_limits(region)

    # download data to appropriate folder(s)
    fetch_weather_data(
        download_dest_dir=save_dir,
        weather_params=weather_params,
        years=years,
        lat_lims=lat_lims,
        lon_lims=lon_lims,
        format="netcdf",
    )

    # combine files in folder to single folder
    combined_save_dir = file_ops.guarantee_existence(save_dir / "weather_parameters")
    print("\n")
    for param in weather_params:
        # get path to unmerged files
        param_dir = save_dir / param
        merged_name = generate_spatiotemporal_var_filename_from_dict(
            {
                "var": param,
                "lats": lat_lims,
                "lons": lon_lims,
                "year": f"{str(years[0])}-{str(years[-1])}",
            }
        )
        # generate combined save path
        combined_save_path = (combined_save_dir / merged_name).with_suffix(".nc")

        file_ops.merge_nc_files_in_dir(
            nc_dir=param_dir, merged_save_path=combined_save_path, format=".netcdf"
        )

    print(
        f"\nAll ERA5 weather files downloaded by year and merged into {combined_save_dir}"
    )

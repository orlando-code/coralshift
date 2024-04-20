# spatial
import geopandas as gpd
import cartopy.crs as ccrs
import rasterio
from rasterio import features as featuresio
import xarray as xa

from pyinterp.backends import xarray
from pyinterp import fill
import xesmf as xe

# general
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import calendar
import pandas as pd
from tqdm.auto import tqdm
from functools import lru_cache

# file handling
from pathlib import Path
import sys
import re

# ml
# from sklearn.metrics import mean_squared_error
# from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor

# from coralshift.machine_learning import baselines
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import xgboost as xgb


# custom
from coralshift.processing import spatial_data
from coralshift.plotting import spatial_plots
from coralshift.machine_learning import baselines

from rasterio.enums import MergeAlg


def rasterize_geodf(
    geo_df: gpd.geodataframe,
    resolution: float = 1.0,
    all_touched: bool = True,
) -> np.ndarray:
    """Rasterize a geodataframe to a numpy array.

    Args:
        geo_df (gpd.geodataframe): Geodataframe to rasterize
        resolution (float): Resolution of the raster in degrees

    Returns:
        np.ndarray: Rasterized numpy array
    TODO: add crs customisation. Probably from class object elsewhere.
    Currently assumes EPSG:4326.
    """

    xmin, ymin, xmax, ymax, width, height = lat_lon_vals_from_geo_df(geo_df, resolution)
    # Create the transform based on the extent and resolution
    transform = rasterio.transform.from_bounds(xmin, ymin, xmax, ymax, width, height)
    transform.crs = rasterio.crs.CRS.from_epsg(4326)

    # Any chance of a loading bar? No: would have to dig into the function istelf.
    # could be interesting...
    return featuresio.rasterize(
        [(shape, 1) for shape in geo_df["geometry"]],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        all_touched=all_touched,
        dtype=rasterio.uint8,
        merge_alg=MergeAlg.add,
    )


def raster_to_xarray(
    raster: np.ndarray,
    x_y_limits: np.ndarray,
    resolution: float = 1.0,
    name: str = "raster_cast_to_xarray",
) -> xa.DataArray:
    """Convert a raster to an xarray DataArray.

    Args:
        raster (np.ndarray): Raster to convert
        resolution (float): Resolution of the raster in degrees

    Returns:
        xa.DataArray: DataArray of the raster
    TODO: add attributes kwarg
    """

    lon_min, lat_min, lon_max, lat_max = x_y_limits
    cell_width = int((lon_max - lon_min) / resolution)
    cell_height = int((lat_max - lat_min) / resolution)

    # Create longitude and latitude arrays
    longitudes = np.linspace(lon_min, lon_max, cell_width)
    # reversed because raster inverted
    latitudes = np.linspace(lat_max, lat_min, cell_height)

    # Create an xarray DataArray with longitude and latitude coordinates
    xa_array = xa.DataArray(
        raster,
        coords={"latitude": latitudes, "longitude": longitudes},
        dims=["latitude", "longitude"],
        name=name,
    )
    # Set the CRS (coordinate reference system) if needed
    # TODO: make kwarg
    xa_array.attrs["crs"] = "EPSG:4326"  # Example CRS, use the appropriate CRS
    # TODO: set attributes if required
    #     attrs=dict(
    #         description="Rasterised Reef Check coral presence survey data"
    #     ))
    return spatial_data.process_xa_d(xa_array)


def lat_lon_vals_from_geo_df(geo_df: gpd.geodataframe, resolution: float = 1.0):
    # Calculate the extent in degrees from bounds of geometry objects
    lon_min, lat_min, lon_max, lat_max = geo_df["geometry"].total_bounds
    # Calculate the width and height of the raster in pixels based on the extent and resolution
    width = int((lon_max - lon_min) / resolution)
    height = int((lat_max - lat_min) / resolution)

    return lon_min, lat_min, lon_max, lat_max, width, height


def rasterise_points_df(
    df: pd.DataFrame,
    lat_column: str,
    lon_column: str,
    resolution: float = 1.0,
    bbox: list[float] = [-90, -180, 90, 180],
) -> np.ndarray:
    """Rasterize a pandas dataframe of points to a numpy array.

    Args:
        df (pd.DataFrame): Dataframe of points to rasterize
        resolution (float): Resolution of the raster in degrees

    Returns:
        np.ndarray: Rasterized numpy array"""

    # extract bbox limits of your raster
    min_lat, min_lon, max_lat, max_lon = bbox

    # Calculate the number of rows and columns in the raster
    num_rows = int((max_lat - min_lat) / resolution)
    num_cols = int((max_lon - min_lon) / resolution)

    # Initialize an empty raster (of zeros)
    raster = np.zeros((num_rows, num_cols), dtype=int)

    # Convert latitude and longitude points to row and column indices of raster
    row_indices = ((max_lat - df[lat_column]) // resolution).astype(int)
    col_indices = ((df[lon_column] - min_lon) // resolution).astype(int)

    # Filter coordinates that fall within the bounding box: this produces a binary mask
    valid_indices = (
        (min_lat <= df[lat_column])
        & (df[lat_column] <= max_lat)
        & (min_lon <= df[lon_column])
        & (df[lon_column] <= max_lon)
    )

    # # Update the raster with counts of valid coordinates
    raster[row_indices[valid_indices], col_indices[valid_indices]] += 1

    # list of row, column indices corresponding to each latitude/longitude point
    valid_coordinates = list(
        zip(row_indices[valid_indices], col_indices[valid_indices])
    )
    # count number of repeated index pairs and return unique
    unique_coordinates, counts = np.unique(
        valid_coordinates, axis=0, return_counts=True
    )
    # assign number of counts to each unique raster
    raster[unique_coordinates[:, 0], unique_coordinates[:, 1]] = counts

    return raster


def try_convert_to_float(value):
    try:
        return float(value)
    except ValueError:
        return np.nan


def check_var_has_coords(
    xa_array, required_coords: list[str] = ["time", "latitude", "longitude"]
):
    """
    Select and return data variables that have 'time', 'latitude', and 'longitude' coordinates.

    Args:
        dataset (xarray.Dataset): Input xarray dataset.

    Returns:
        xarray.Dataset: Subset of the input dataset containing selected data variables.

    N.B. must have consistent naming of coordinate dimensions by this point.
    """
    if all(coord in xa_array.coords for coord in required_coords):
        return True
    else:
        return False


def apply_fill_loess(dataset: xa.Dataset, nx=2, ny=2):
    """
    Apply fill.loess to each time step for each variable in the xarray dataset.

    Args:
        dataset (xarray.Dataset): Input xarray dataset with time series of variables.
        nx (int): Number of pixels to extend in the x-direction.
        ny (int): Number of pixels to extend in the y-direction.

    Returns:
        xarray.Dataset: Buffered xarray dataset.
    """
    # TODO: nested tqdm in notebooks and scripts
    # Create a copy of the original dataset
    buffered_dataset = dataset.copy(deep=True)
    buffered_data_vars = buffered_dataset.data_vars

    print(f"{len(buffered_data_vars)} raster(s) to spatially buffer...")
    for _, (var_name, var_data) in tqdm(
        enumerate(buffered_data_vars.items()),
        desc="Buffering variables",
        total=len(buffered_data_vars),
        position=0,
    ):  # for each variable in the dataset
        if check_var_has_coords(
            var_data
        ):  # if dataset has latitude, longitude, and time coordinates
            for t in tqdm(
                buffered_dataset.time,
                desc=f"Processing timesteps of variable '{var_name}'",
                leave=False,
                position=1,
            ):  # buffer each timestep
                grid = xarray.Grid2D(var_data.sel(time=t))
                filled = fill.loess(grid, nx=nx, ny=ny)
                buffered_data_vars[var_name].loc[dict(time=t)] = filled.T
        elif check_var_has_coords(
            var_data, ["latitude", "longitude"]
        ):  # if dataset has latitude, longitude only
            # for var_name in tqdm(list(buffered_dataset.data_vars.keys()), desc="Processing variables..."):  #
            grid = xarray.Grid2D(var_data, geodetic=False)
            filled = fill.loess(grid, nx=nx, ny=ny)
            # buffered_data_vars[var_name].loc[dict(var=var)] = filled.T
            buffered_dataset.update(
                {var_name: (buffered_dataset[var_name].dims, filled)}
            )
        else:
            print(
                f"""Variable must have at least 'latitude', and 'longitude' coordinates to be spatially padded.
                \nVariable {var_name} has '{var_data.coords}'. Skipping..."""
            )

    return buffered_dataset


def resample_xa_d(
    xa_d: xa.DataArray | xa.Dataset,
    lat_range: list[float] = None,
    lon_range: list[float] = None,
    resolution: float = 0.1,
    resample_method: str = "linear",
):
    """
    Resample an xarray DataArray or Dataset to a common extent and resolution.

    Args:
        xa_d (xa.DataArray | xa.Dataset): xarray DataArray or Dataset to resample.
        lat_range (list[float]): Latitude range of the common extent.
        lon_range (list[float]): Longitude range of the common extent.
        resolution (float): Longitude resolution of the common extent.
        resample_method (str, optional): Resampling method to use. Defaults to "linear".

    Returns:
        xa.DataArray | xa.Dataset: Resampled xarray DataArray or Dataset.

    resample_methods:
        "linear" – Bilinear interpolation.
        "nearest" – Nearest-neighbor interpolation.
        "zero" – Piecewise-constant interpolation.
        "slinear" – Spline interpolation of order 1.
        "quadratic" – Spline interpolation of order 2.
        "cubic" – Spline interpolation of order 3.
        TODO: implement"polynomial"
    """
    # if coordinate ranges not specified, infer from present xa_d
    if not (lat_range and lon_range):
        lat_range = spatial_data.min_max_of_coords(xa_d, "latitude")
        lon_range = spatial_data.min_max_of_coords(xa_d, "longitude")

    lat_range = sorted(lat_range)
    lon_range = sorted(lon_range)

    # Create a dummy dataset with the common extent and resolution
    common_dataset = xa.Dataset(
        coords={
            "latitude": (
                ["latitude"],
                np.arange(lat_range[0], lat_range[1] + resolution, resolution),
            ),
            "longitude": (
                ["longitude"],
                np.arange(lon_range[0], lon_range[1] + resolution, resolution),
            ),
        }
    )
    current_resolution = get_resolution(xa_d)
    # if upsampling, interpolate
    if resolution < current_resolution:
        return xa_d.sel(
            latitude=slice(*lat_range), longitude=slice(*lon_range)
        ).interp_like(common_dataset, method=resample_method)
    # if downsampling, coarsen
    else:
        return coarsen_xa_d(
            xa_d.sel(latitude=slice(*lat_range), longitude=slice(*lon_range)),
            resolution,
            resample_method,
        )


# @lru_cache(maxsize=None)  # Set maxsize to limit the cache size, or None for unlimited
def xesmf_regrid(
    xa_d: xa.DataArray | xa.Dataset,
    lat_range: list[float] = None,
    lon_range: list[float] = None,
    resolution: float = 0.1,
    resampled_method: str = "bilinear",
):

    lon_range = sorted(lon_range)
    lat_range = sorted(lat_range)
    target_grid = xe.util.grid_2d(
        lon_range[0],
        lon_range[1],
        resolution,
        lat_range[0],
        lat_range[1],
        resolution,  # longitude range and resolution
    )  # latitude range and resolution

    regridder = xe.Regridder(xa_d, target_grid, method=resampled_method)

    # return spatial_data.process_xa_d(regridder(xa_d))
    return process_xesmf_regridded(regridder(xa_d))


def process_xesmf_regridded(
    xa_d: xa.DataArray | xa.Dataset,
):
    xa_d["lon"] = xa_d.lon.values[0, :]
    xa_d["lat"] = xa_d.lat.values[:, 0]

    return xa_d.rename(
        {"x": "longitude", "y": "latitude", "lon": "longitude", "lat": "latitude"}
    )


def coarsen_xa_d(xa_d, resolution: float = 0.1, method="mean"):
    # TODO: for now, treating lat and long with indifference (since this is how data is).
    num_points_lat = int(
        round(resolution / abs(xa_d["latitude"].diff("latitude").mean().values))
    )
    num_points_lon = int(
        round(resolution / abs(xa_d["longitude"].diff("longitude").mean().values))
    )

    return xa_d.coarsen(
        latitude=num_points_lat,
        longitude=num_points_lon,
        boundary="pad",
    ).reduce(method)


def get_resolution(xa_d: xa.Dataset | xa.DataArray) -> float:
    """
    Get the resolution of an xarray dataset or data array.

    Parameters:
    - xa_d: The xarray dataset or data array.

    Returns:
    - The resolution of the xarray dataset or data array.
    """
    if "latitude" in xa_d.coords:
        lat_diff = xa_d.latitude.diff(dim="latitude").values[0]
    elif "lat" in xa_d.coords:
        lat_diff = xa_d.lat.diff(dim="lat").values[0]
    else:
        raise ValueError(
            "Latitude coordinate not found in the xarray dataset or data array."
        )

    if "longitude" in xa_d.coords:
        lon_diff = xa_d.longitude.diff(dim="longitude").values[0]
    elif "lon" in xa_d.coords:
        lon_diff = xa_d.lon.diff(dim="lon").values[0]
    else:
        raise ValueError(
            "Longitude coordinate not found in the xarray dataset or data array."
        )

    resolution = min(abs(lat_diff), abs(lon_diff))
    return resolution


def resample_to_other(
    xa_d_to_resample, target_xa, resample_method: str = "linear"
) -> xa.Dataset | xa.DataArray:
    """
    Resample an xarray DataArray or Dataset to the resolution and extent of another xarray DataArray or Dataset.

    Args:
        xa_d_to_resample (xa.DataArray | xa.Dataset): xarray DataArray or Dataset to resample.
        target_xa (xa.DataArray | xa.Dataset): xarray DataArray or Dataset to resample to.

    Returns:
        xa.DataArray | xa.Dataset: Resampled xarray DataArray or Dataset.

    resample_methods:
        "linear" – Bilinear interpolation.
        "nearest" – Nearest-neighbor interpolation.
        "zero" – Piecewise-constant interpolation.
        "slinear" – Spline interpolation of order 1.
        "quadratic" – Spline interpolation of order 2.
        "cubic" – Spline interpolation of order 3.
        TODO: implement"polynomial"
    """
    return xa_d_to_resample.interp_like(target_xa, method=resample_method)


def spatially_combine_xa_d_list(
    xa_d_list: list[xa.DataArray | xa.Dataset],
    lat_range: list[float],
    lon_range: list[float],
    resolution: float,
    resample_method: str = "linear",
) -> xa.Dataset:
    """
    Resample and merge a list of xarray DataArrays or Datasets to a common extent and resolution.

    Args:
        xa_d_list (list[xa.DataArray | xa.Dataset]): List of xarray DataArrays or Datasets to resample and merge.
        lat_range (list[float]): Latitude range of the common extent.
        lon_range (list[float]): Longitude range of the common extent.
        resolution (float): Resolution of the common extent.
        resample_method (str, optional): Resampling method to use. Defaults to "linear".

    Returns:
        xa.Dataset: Dataset containing the resampled and merged DataArrays or Datasets.

    N.B. resample_method can take the following values for 1D interpolation:
        - "linear": Linear interpolation.
        - "nearest": Nearest-neighbor interpolation.
        - "zero": Piecewise-constant interpolation.
        - "slinear": Spline interpolation of order 1.
        - "quadratic": Spline interpolation of order 2.
        - "cubic": Spline interpolation of order 3.
    and these for n-d interpolation:
        - "linear": Linear interpolation.
        - "nearest": Nearest-neighbor interpolation.
    """

    # Create a new dataset with the common extent and resolution
    common_dataset = xa.Dataset(
        coords={
            "latitude": (["latitude"], np.arange(*np.sort(lat_range), resolution)),
            "longitude": (
                ["longitude"],
                np.arange(*np.sort(lon_range), resolution),
            ),
        }
    )

    # Iterate through input datasets, resample, and merge into the common dataset
    for input_ds in tqdm(
        xa_d_list, desc=f"resampling and merging {len(xa_d_list)} datasets"
    ):
        # Resample the input dataset to the common resolution and extent using bilinear interpolation
        resampled_dataset = input_ds.interp(
            latitude=common_dataset["latitude"].sel(
                latitude=slice(min(lat_range), max(lat_range))
            ),
            longitude=common_dataset["longitude"].sel(
                longitude=slice(min(lon_range), max(lon_range))
            ),
            method=resample_method,
        )

        # Merge the resampled dataset into the common dataset
        common_dataset = xa.merge(
            [common_dataset, resampled_dataset], compat="no_conflicts"
        )

    return common_dataset


def train_test_val_split(
    df_X: pd.DataFrame,
    df_y: pd.Series,
    ttv_fractions: list[float],
    split_method: str = "pixelwise",
    orientation: str = "vertical",
    random_state: int = 42,
):
    if split_method == "pixelwise":
        train_X, test_val_X, train_y, test_val_y = train_test_split(
            df_X, df_y, test_size=sum(ttv_fractions[1:]), random_state=random_state
        )
        test_size = ttv_fractions[1] / sum(ttv_fractions[1:])
        if test_size == 1:
            test_X, test_y = test_val_X, test_val_y
            # TODO: make less hacky (currently just copying over the same values), even tho val specified as 0
            val_X, val_y = test_val_X, test_val_y
        else:
            test_X, val_X, test_y, val_y = train_test_split(
                test_val_X, test_val_y, test_size=test_size, random_state=random_state
            )
        return (
            (train_X, train_y),
            (test_X, test_y),
            (val_X, val_y),
        )
    elif split_method == "spatial":
        # TODO: correct/check
        return ttv_spatial_split(
            pd.concat([df_X, df_y], axis=1), ttv_fractions, orientiation=orientation
        )


def ttv_spatial_split(
    df: pd.DataFrame, ttv_fractions: list[float], orientation: str = "vertical"
) -> list[pd.DataFrame]:
    """
    Splits a dataframe into train, test, and validation sets, spatially.

    Args:
        df (pd.DataFrame): The dataframe to split.
        ttv_fractions (list[float]): The fractions of the dataframe to allocate to train, test, and validation sets.
        orientation (str, optional): The orientation of the splits. Defaults to "vertical".

    Returns:
        list[pd.DataFrame]: The train, test, and validation sets.
    """
    # check that ttv_fractions sum to 1 or if any is zero
    assert (
        sum(ttv_fractions) == 1 or 0 in ttv_fractions
    ), f"ttv_fractions must sum to 1 or contain a zero. Currently sum to {sum(ttv_fractions)}"

    assert orientation in [
        "vertical",
        "horizontal",
    ], f"orientation must be 'vertical' or 'horizontal'. Currently {orientation}"

    if orientation == "horizontal":
        df = df.sort_values(by="latitude")
    elif orientation == "vertical":
        df = df.sort_values(by="longitude")

    df_list = []

    # this was hanlding omission of val completely: for now, keeping it in, but just empty
    # if 0 in ttv_fractions:
    #     nonzero_fractions = [frac for frac in ttv_fractions if frac != 0]
    #     split_indices = [int(frac * len(df)) for frac in np.cumsum(nonzero_fractions)]

    #     for idx, split_idx in enumerate(split_indices):
    #         if idx == 0:
    #             df_list.append(df.iloc[:split_idx])
    #         elif idx == len(split_indices) - 1:
    #             df_list.append(df.iloc[split_idx:])
    #         else:
    #             df_list.append(df.iloc[split_indices[idx - 1] : split_idx])
    # else:
    df_list = np.split(
        df,
        [
            int(ttv_fractions[0] * len(df)),
            int((ttv_fractions[0] + ttv_fractions[1]) * len(df)),
        ],
    )

    return df_list


def onehot_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode the NaN values of a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.

    Returns:
        pd.DataFrame: The one-hot encoded DataFrame.
    """
    # Create a copy of the original DataFrame to avoid modifying it in place
    # TODO: will this lead to a memory issue for large dfs?
    onehot_df = df.copy()
    # create a "onehot_nan" column: 0 if nans present in row, 1 otherwise
    onehot_df["onehot_nan"] = df.isna().any(axis=1).astype(int)
    # all nans to 0
    # TODO: is this the best way to deal with nans?
    onehot_df.fillna(value=0, inplace=True)

    return onehot_df


def scaling_params_from_df(df: pd.DataFrame, var_cols: list[str]) -> dict:
    """
    Calculate scaling parameters (min, max) for the specified variable columns in a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        var_cols (list[str]): List of column names to calculate scaling parameters for.

    Returns:
        dict: Dictionary with column names as keys and (min, max) tuples as values.

    TODO: expand to other scaling methods
    TODO: useful to have values as list rather than dict?
    """
    scaling_params = df[var_cols].agg(["min", "max"], skipna=True)
    # cast from dataframe to dictionary. Keys: column names, values: (min, max) tuples
    scaling_params_dict = scaling_params.to_dict()

    return scaling_params_dict


# scale dataframe using scaling_params
def scale_dataframe(df: pd.DataFrame, scaling_params: dict) -> pd.DataFrame:
    """
    Scale the specified columns of a DataFrame using the provided scaling parameters.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        scaling_params (dict): Dictionary with column names as keys and (min, max) scaling parameters as values.

    Returns:
        pd.DataFrame: The scaled DataFrame.
    """
    # Create a copy of the original DataFrame to avoid modifying it in place
    scaled_df = df.copy()

    for col, params in scaling_params.items():
        min_val = params.get("min", 0.0)
        max_val = params.get("max", 1.0)
        scaled_df[col] = (scaled_df[col] - min_val) / (max_val - min_val)

    return scaled_df


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


def year_to_datetime(year: int, xa_d: xa.DataArray | xa.Dataset = None) -> datetime:
    """
    Convert an integer denoting a year to a datetime object.

    Args:
        year (int): The year to convert.
        xa_d (xa.DataArray | xa.Dataset, optional): An xarray DataArray or Dataset to check the year against.
        Defaults to None.

    Returns:
        datetime: The datetime object corresponding to the year.
    """
    # TODO: replace with datetime module
    if xa_d:
        # Extract the minimum and maximum years from the dataset's time coordinate
        min_year, max_year = spatial_data.min_max_of_coords(xa_d, "time")

        # Ensure the provided year is within the range of the dataset's time coordinate
        if year < min_year or year > max_year:
            raise ValueError(f"Year {year} is outside the dataset's time range")

    return datetime(year, 1, 1)


import warnings


def calculate_statistics(
    xa_ds: xa.Dataset,
    vars: list[str] = ["so", "thetao", "tos", "uo", "vo"],
    years_window: tuple[int] = None,
) -> xa.Dataset:
    """
    Calculate statistics for each variable in the dataset, similar to Couce (2012, 2023).

    Args:
        xa_ds (xa.Dataset): Input xarray dataset.
        vars (list[str], optional): List of variable names to calculate statistics for.
        Defaults to ["so", "thetao", "tos", "uo", "vo"].
        years_window (tuple[int], optional): The time period to calculate statistics for. Defaults to None.

    Returns:
        xa.Dataset: Dataset containing the calculated statistics.
    """
    if years_window:
        # Select the time period of interest
        xa_ds = xa_ds.sel(
            time=slice(
                year_to_datetime(min(years_window)), year_to_datetime(max(years_window))
            )
        )

    stats = {}

    for i, var_name in tqdm(
        enumerate(vars), desc="calculating statistics for variables", total=len(vars)
    ):
        var_data = xa_ds[var_name]
        # Calculate annual average
        # annual_mean = var_data.resample(time='1Y').mean()
        # stats[f"{var_name}_am"] = annual_mean

        # Calculate mean for each month
        monthly_mean = var_data.groupby("time.month").mean(dim="time")

        # Map numerical month values to month names
        month_names = [calendar.month_name[i] for i in monthly_mean["month"].values]

        # Assign monthly means to their respective month names
        for i, month in enumerate(month_names):
            stats[f"{var_name}_{month.lower()}_mean"] = monthly_mean.isel(
                month=i
            ).values

        # Calculate maximum and minimum of monthly values over the whole time period
        monthly_max_overall = var_data.groupby("time.month").max(dim="time")
        monthly_min_overall = var_data.groupby("time.month").min(dim="time")

        for i, month in enumerate(month_names):
            stats[f"{var_name}_{month.lower()}_max"] = monthly_max_overall.isel(
                month=i
            ).values
            stats[f"{var_name}_{month.lower()}_min"] = monthly_min_overall.isel(
                month=i
            ).values

        # Calculate standard deviation of time steps
        time_std = var_data.std(dim="time", skipna=None)
        stats[f"{var_name}_time_std"] = time_std.values

        # Calculate standard deviation of January and July values
        january_std = var_data.where(var_data["time.month"] == 1).std(dim="time")
        july_std = var_data.where(var_data["time.month"] == 7).std(dim="time")
        stats[f"{var_name}_jan_std"] = january_std.values
        stats[f"{var_name}_jul_std"] = july_std.values

        # Calculate the overall mean for each statistic
        stats[f"{var_name}_overall_mean"] = var_data.mean(dim="time").values

    # Combine all calculated variables into a new dataset, retaining the original dataset's attributes, coordinates etc.
    # stats_xa = xa.Dataset(stats)

    stats_xa = xa.Dataset(
        {key: (("latitude", "longitude"), value) for key, value in stats.items()},
        coords={"latitude": var_data.latitude, "longitude": var_data.longitude},
    )

    stats_xa.attrs = xa_ds.attrs

    return stats_xa


def get_min_max_coords(ds, coord):
    min_coord = float(min(ds[coord]).values)
    max_coord = float(max(ds[coord]).values)
    return min_coord, max_coord


def investigate_depth_mask(comp_var_xa, mask_var_xa, var_limits: list[tuple[float]]):
    # TODO: make less janky
    raw_vals = []
    positive_negative_ratio = []
    x_labels = [str(lim_pair) for lim_pair in var_limits]

    for lim_pair in var_limits:
        masked_vals = generate_var_mask(comp_var_xa, mask_var_xa, lim_pair)
        val_sum = np.nansum(masked_vals["elevation"].values)
        raw_vals.append(val_sum)

        positive_negative_ratio.append(
            val_sum / np.count_nonzero(masked_vals["elevation"].values == 0)
        )

    return raw_vals, positive_negative_ratio


def generate_var_mask(
    comp_var_xa: xa.DataArray,
    mask_var_xa: xa.DataArray,
    limits: tuple[float] = [-2000, 0],
) -> xa.DataArray:
    mask = (mask_var_xa <= max(limits)) & (mask_var_xa >= min(limits))
    return comp_var_xa.where(mask, drop=True)


def iterative_to_string_list(iter_obj: tuple, dp: int = 0):
    # Round the values in the iterable object to the specified number of decimal places
    return [round(i, dp) for i in iter_obj]


def lat_lon_string_from_tuples(
    lats: tuple[float, float], lons: tuple[float, float], dp: int = 0
):
    round_lats = iterative_to_string_list(lats, dp)
    round_lons = iterative_to_string_list(lons, dp)

    return (
        f"n{max(round_lats)}_s{min(round_lats)}_w{min(round_lons)}_e{max(round_lons)}"
    )


# PROBLEM WITH DOUBLE NEGATIVES
# def lat_lon_string_from_tuples(
#     lats: tuple[float, float], lons: tuple[float, float], dp: int = 0
# ):
#     round_lats = iterative_to_string_list(lats, dp)
#     round_lons = iterative_to_string_list(lons, dp)

#     return f"lats_{min(round_lats)}-{max(round_lats)}_lons_{min(round_lons)}-{max(round_lons)}"


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


def find_files_for_area(filepaths, lat_range, lon_range):
    result = []

    for filepath in filepaths:
        fp_lats, fp_lons = extract_lat_lon_ranges_from_fp(filepath)

        if (
            max(lat_range) <= max(fp_lats)
            and min(lat_range) >= min(fp_lats)
            and max(lon_range) <= max(fp_lons)
            and min(lon_range) >= min(fp_lons)
        ):
            result.append(filepath)

    return result


def split_dataset_and_save(
    ds_fp, divisor, output_dir_name: str = None, select_vars: list[str] = None
):
    ds = xa.open_dataset(ds_fp)

    if select_vars:
        ds = ds[select_vars]

    subsets_dict = split_dataset_by_indices(ds, divisor)

    # Create a subdirectory to save the split datasets
    if output_dir_name:
        output_dir = (
            Path(ds_fp).parent / f"{output_dir_name}_{divisor**2}_split_datasets"
        )
    else:
        output_dir = Path(ds_fp).parent / f"{divisor**2}_split_datasets"
    output_dir.mkdir(parents=True, exist_ok=True)

    for coord_info, subset in tqdm(
        subsets_dict.items(), desc="saving dataset subsets..."
    ):
        stem_stem = str(Path(ds_fp).stem).split("lats")[0]
        # Construct the filename based on bounds
        filename = f"{stem_stem}_{coord_info}.nc"
        save_fp = output_dir / filename
        subset.to_netcdf(save_fp)
    return subsets_dict


def cont_to_class(array, threshold=0.5):
    """Return thresholded values, leaving nans untouched."""
    # N.B. recently modified. Used extensively in model preparation, so be on lookout.
    # array = array.copy()

    # array[array > threshold] = 1
    # array[array <= threshold] = 0

    # return array.astype(int)
    return np.where(np.isnan(array), np.nan, np.where(array >= threshold, 1, 0)).astype(
        int
    )


def split_dataset_by_indices(dataset, divisor) -> dict:
    subsets_dict = {}
    num_lats = len(dataset.latitude.values) // divisor
    num_lons = len(dataset.longitude.values) // divisor
    for i in range(divisor):
        for j in range(divisor):
            start_lat_ind = i * num_lats
            start_lon_ind = j * num_lons

            subset = dataset.isel(
                latitude=slice(start_lat_ind, start_lat_ind + num_lats),
                longitude=slice(start_lon_ind, start_lon_ind + num_lons),
            )

            lat_lims = spatial_data.min_max_of_coords(subset, "latitude")
            lon_lims = spatial_data.min_max_of_coords(subset, "longitude")

            coord_info = lat_lon_string_from_tuples(lat_lims, lon_lims)
            subsets_dict[coord_info] = subset

    return subsets_dict


def ds_to_ml_ready(
    xa_ds: xa.Dataset,
    predictand: str = "UNEP_GDCR",
    pos_neg_ratio: float = 0.1,
    depth_mask_lims: tuple[float, float] = [-50, 0],
    exclude_list: list[str] = [
        "latitude",
        "longitude",
        "latitude_grid",
        "longitude_grid",
        "crs",
        "depth",
        "spatial_ref",
    ],
    remove_rows: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Convert an xarray Dataset to a format suitable for machine learning.

    Args:
        xa_ds (xa.Dataset): The xarray Dataset to convert.
        predictand (str, optional): The name of the ground truth variable. Defaults to "UNEP_GDCR".
        pos_neg_ratio (float, optional): The ratio of positive to negative samples for classification. Defaults to 0.1.
        depth_mask_lims (tuple[float, float], optional): The depth limits to use for masking. Defaults to [-50, 0].
        exclude_list (list[str], optional): List of variables to exclude from the conversion.
            Defaults to ["latitude", "longitude", "latitude_grid", "longitude_grid", "crs", "depth", "spatial_ref"].
        remove_rows (bool, optional): Whether to remove rows beyond depth limits. Defaults to True.

    Returns:
        tuple[pd.DataFrame, pd.Series]: The converted features (X) and target variable (y).
    """
    # de-dask and convert to dataframe
    df = xa_ds.compute().to_dataframe()
    # TODO: implement checking for empty dfs

    predictors = [
        pred for pred in df.columns if pred != predictand and pred not in exclude_list
    ]

    df_masked = adaptive_depth_mask(
        df,
        depth_mask_lims=depth_mask_lims,
        pos_neg_ratio=pos_neg_ratio,
        remove_rows=remove_rows,
        predictand=predictand,
        depth_var="elevation",
    )

    # encode any rows containing nans to additional column
    df_masked["nan_onehot"] = df_masked.isna().any(axis=1).astype(int)
    # fill any nans with zeros
    df_masked = df_masked.fillna(0)

    X = df_masked[predictors]
    y = df_masked[predictand]

    return X, y


def calc_non_zero_ratio(df, predictand=None):
    # if df a series
    if isinstance(df, pd.Series):
        return np.where(df > 0, 1, 0).sum() / len(df)
    else:
        return np.where(df[predictand] > 0, 1, 0).sum() / len(df)


def adaptive_depth_mask(
    df,
    depth_mask_lims=[-50, 0],
    pos_neg_ratio=0.1,
    tolerance=0.005,
    remove_rows=True,
    predictand="UNEP_GDCR",
    depth_var="elevation",
):
    hold_df = df.copy()
    depth_mask_lims = sorted(depth_mask_lims)
    # where value in df is not zero
    non_zero_ratio = calc_non_zero_ratio(df, predictand)
    # print(non_zero_ratio)

    best_non_zero_ratio_diff = 1
    # Iterate for a fixed number of times
    for i in range(1000):
        prev_df = df.copy()  # Store previous depth_mask_lims

        if (
            non_zero_ratio >= pos_neg_ratio - tolerance
            and non_zero_ratio <= pos_neg_ratio + tolerance
        ):
            return df  # Exit the loop if pos_neg_ratio is within tolerance

        # TODO: adjust depending on depth_mask_lims_size?
        if non_zero_ratio > pos_neg_ratio + tolerance:
            # increase minimum depth
            depth_mask_lims[0] -= 1
        elif non_zero_ratio < pos_neg_ratio - tolerance:
            # decrease minimum depth
            depth_mask_lims[0] += 1

        df = depth_filter(df, depth_mask_lims, remove_rows, depth_var)
        non_zero_ratio = calc_non_zero_ratio(df, predictand)

        # Check for nan non_zero_ratio
        if np.isnan(non_zero_ratio):
            df = prev_df  # Restore previous depth_mask_lims
            break

        pos_neg_non_zero_diff = abs(pos_neg_ratio - non_zero_ratio)
        if pos_neg_non_zero_diff < best_non_zero_ratio_diff:
            best_non_zero_ratio_diff = pos_neg_non_zero_diff
            best_ratio_depth_mask_lims = depth_mask_lims.copy()

        print(f"{i} depth mask", depth_mask_lims)
        print("pos_neg_ratio", non_zero_ratio)

    print(best_ratio_depth_mask_lims)
    # if loop not satisfied, return depth_mask_lims which get closest to pos_neg_ratio
    return depth_filter(hold_df, best_ratio_depth_mask_lims, remove_rows, depth_var)


def depth_filter(
    df: pd.DataFrame,
    depth_mask_lims: list[float, float],
    remove_rows: bool = True,
    depth_var: str = "elevation",
):
    df_depth = df.copy()
    # generate boolean depth mask
    depth_condition = (df_depth[depth_var] < max(depth_mask_lims)) & (
        df_depth[depth_var] > min(depth_mask_lims)
    )
    # if remove_rows (default), remove rows outside of depth mask
    if remove_rows:
        df_depth = df_depth[depth_condition]
    # if not remove_rows, add a column of 1s and 0s to indicate whether row is within depth mask
    else:
        df_depth["within_depth"] = 0
        df_depth.loc[depth_condition, "within_depth"] = 1

    return df_depth


def customize_plot_colors(fig, ax, background_color="#212121", text_color="white"):
    # Set figure background color
    fig.patch.set_facecolor(background_color)

    # Set axis background color (if needed)
    ax.set_facecolor(background_color)

    # Set text color for all elements in the plot
    for text in fig.texts:
        text.set_color(text_color)
    for text in ax.texts:
        text.set_color(text_color)
    for text in ax.xaxis.get_ticklabels():
        text.set_color(text_color)
    for text in ax.yaxis.get_ticklabels():
        text.set_color(text_color)
    ax.title.set_color(text_color)
    ax.xaxis.label.set_color(text_color)
    ax.yaxis.label.set_color(text_color)

    # Set legend text color
    legend = ax.get_legend()
    if legend:
        for text in legend.get_texts():
            text.set_color(text_color)
    # # set cbar labels
    # cbar = ax.collections[0].colorbar
    # cbar.set_label(color=text_color)
    # cbar.ax.yaxis.label.set_color(text_color)
    ax.tick_params(axis="x", colors="white")
    ax.tick_params(axis="y", colors="white")

    return fig, ax


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


def generate_remap_info(eg_nc, resolution=0.25, gridtype: str = "lonlat"):
    xsize = 360 / resolution
    ysize = 180 / resolution

    xfirst = float(np.min(eg_nc.longitude).values)
    yfirst = float(np.min(eg_nc.latitude).values)

    x_inc, y_inc = resolution, resolution
    return xsize, ysize, xfirst, yfirst, x_inc, y_inc


def pd_to_array(df):
    """Converts pandas dataframe to xarray instance"""
    return df.to_xarray().sortby("longitude").sortby("latitude")


def crop_df(df, lat_lims, lon_lims):
    """Crops dataframe to specified lat and lon limits"""
    return df[
        (df.index.get_level_values("latitude") >= min(lat_lims))
        & (df.index.get_level_values("latitude") <= max(lat_lims))
        & (df.index.get_level_values("longitude") >= min(lon_lims))
        & (df.index.get_level_values("longitude") <= max(lon_lims))
    ]


def visualise_train_test_points(
    y_train_df: pd.DataFrame | pd.Series,
    y_test_df: pd.DataFrame | pd.Series,
    do_crop_df: bool = False,
    lat_lims=[-10, -13],
    lon_lims=[147, 149],
    sample_frequency: int = -1,
):
    """
    Visualise the distribution of coral cover values in the train and test sets.

    Args:
        y_train_df (pd.DataFrame | pd.Series): The train set.
        y_test_df (pd.DataFrame | pd.Series): The test set.
        do_crop_df (bool, optional): Whether to crop the dataframes to the specified lat and lon limits.
            Defaults to False.
        lat_lims (list[float], optional): The latitude limits to crop to. Defaults to [-10, -13].
        lon_lims (list[float], optional): The longitude limits to crop to. Defaults to [147, 149].
        sample_frequency (int, optional): The frequency to sample the cropped dataframes at. Defaults to -1.
    """
    # TODO: more customisable plotting
    # combine train and test sets
    comb = pd.concat([y_train_df, y_test_df], axis=1)
    comb["summed"] = comb.fillna(0).sum(axis=1)

    if do_crop_df:
        y_train_plot_df = crop_df(y_train_df, lat_lims, lon_lims)[::sample_frequency]
        y_test_plot_df = crop_df(y_test_df, lat_lims, lon_lims)[::sample_frequency]
        comb_plot_df = crop_df(comb["summed"], lat_lims, lon_lims)[::sample_frequency]

    fig = plt.figure(figsize=(12, 6))
    fig.tight_layout()
    gs = fig.add_gridspec(2, 3)

    plot_titles = ["Train set", "Test set", "All datapoints"]
    hist_colors = ["#3B9AB2", "#EBCC2A", "#d83c04"]
    for i, array in enumerate([y_train_plot_df, y_test_plot_df, comb_plot_df]):
        ax = fig.add_subplot(gs[0, i], projection=ccrs.PlateCarree())
        xa_set = pd_to_array(array)  # create xarray instance
        spatial_plots.plot_spatial(
            xa_set, fax=(fig, ax), cbar=False, title=plot_titles[i]
        )

        ax = fig.add_subplot(gs[1, i])
        ax.set_yscale("log")
        ax.hist(array.values.flatten(), color=hist_colors[i])
        if i == 0:
            ax.set_ylabel("Count (log scale)")

    # superpose train and test sets on all datapoints
    ax.hist(y_train_plot_df.values.flatten(), color=hist_colors[0], alpha=1)
    ax.hist(y_test_plot_df.values.flatten(), color=hist_colors[1], alpha=1)

    # Add a single label to the bottom center of the entire plot
    fig.text(
        0.5,
        0.05,
        "Fraction coral cover",
        ha="center",
        va="center",
        fontsize=12,
        color="black",
    )


def sizeof_fmt(num, suffix="B"):
    """by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified"""
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, "Yi", suffix)


def print_var_mems(num_show):
    for name, size in sorted(
        ((name, sys.getsizeof(value)) for name, value in list(locals().items())),
        key=lambda x: -x[1],
    )[:num_show]:
        print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))


class ModelInitializer:
    def __init__(self, model_type: str, random_state: int = 42, params: dict = None):
        self.random_state = random_state
        self.model_info = [
            # discrete models
            {
                "model_type": "logreg",
                "model_type_full_name": "Logistic Regression",
                "data_type": "discrete",
                "model": LogisticRegression(
                    class_weight="balanced",
                    n_jobs=-1,
                    verbose=1,
                    random_state=self.random_state,
                ),
                "search_grid": baselines.logreg_search_grid(),
            },
            {
                "model_type": "rf_cf",
                "model_type_full_name": "Random Forest Classifier",
                "data_type": "discrete",
                "model": RandomForestClassifier(
                    class_weight="balanced",
                    n_jobs=-1,
                    verbose=1,
                    random_state=self.random_state,
                ),
                "search_grid": baselines.rf_search_grid(),
            },
            {
                "model_type": "gb_cf",
                "model_type_full_name": "Gradient Boosting Classifier",
                "data_type": "discrete",
                "model": GradientBoostingClassifier(
                    verbose=1,
                    random_state=self.random_state,  # N.B. has sample weight rather than class weight. TODO: implement
                ),
                "search_grid": baselines.boosted_search_grid(),
            },
            {
                "model_type": "rf_reg",
                "model_type_full_name": "Random Forest Regressor",
                "data_type": "continuous",
                "model": RandomForestRegressor(
                    verbose=1, n_jobs=-1, random_state=self.random_state
                ),
                "search_grid": baselines.rf_search_grid(),
            },
            {
                "model_type": "gb_reg",
                "model_type_full_name": "Gradient Boosting Regressor",
                "data_type": "continuous",
                "model": GradientBoostingRegressor(
                    verbose=1, random_state=self.random_state
                ),
                "search_grid": baselines.boosted_search_grid(model_type="regressor"),
            },
            {
                "model_type": "xgb_cf",
                "model_type_full_name": "XGBoost Classifier",
                "data_type": "discrete",
                "model": xgb.XGBClassifier(verbose=1, random_state=self.random_state),
                "search_grid": baselines.xgb_search_grid(),
            },
            {
                "model_type": "xgb_reg",
                "model_type_full_name": "XGBoost Regressor",
                "data_type": "continuous",
                "model": xgb.XGBRegressor(verbose=1, random_state=self.random_state),
                "search_grid": baselines.xgb_search_grid(model_type="regressor"),
            },
            {
                "model_type": "mlp_cf",
                "model_type_full_name": "MLP Classifier",
                "data_type": "discrete",
                "model": MLPClassifier(
                    verbose=1, random_state=self.random_state, max_fun=15000
                ),
                "search_grid": baselines.mlp_search_grid(),
            },
            {
                "model_type": "mlp_reg",
                "model_type_full_name": "MLP Regressor",
                "data_type": "continuous",
                "model": MLPRegressor(
                    verbose=1, random_state=self.random_state, max_fun=15000
                ),
                "search_grid": baselines.mlp_search_grid(),
            },
        ]
        self.model_type = self._normalize_model_name(model_type)
        self.params = params

    def _normalize_model_name(self, model_name):
        normalized_names = {
            info["model_type_full_name"]: info["model_type"] for info in self.model_info
        }
        return normalized_names.get(model_name, model_name)

    def get_data_type(self):
        for m in self.model_info:
            if m["model_type"] == self.model_type:
                return m["data_type"]
        else:
            raise ValueError(f"'{self.model_type}' not a valid model.")

    def get_model(self):
        for m in self.model_info:
            if m["model_type"] == self.model_type:
                model_instance = m["model"]
                if self.params:
                    model_instance.set_params(**self.params)
                return model_instance
        else:
            raise ValueError(f"'{self.model_type}' not a valid model.")

    def get_random_search_grid(self):
        for m in self.model_info:
            if m["model_type"] == self.model_type:
                return m["search_grid"]
        else:
            raise ValueError(f"'{self.model_type}' not a valid model.")


# import pymc as pm

# def hierarchical_beta_model():


def calculate_class_weight(label_array: np.ndarray):
    unique_values, counts = np.unique(label_array, return_counts=True)
    occurrence_dict = dict(zip(unique_values, counts))
    return occurrence_dict


# def generate_gridsearch_parameter_grid(params_dict: dict, num_samples: int = 2) -> dict:
#     grid_params = {}
#     for key, value in params_dict.items():
#         if key == "verbose":
#             grid_params[key] = [value]
#         elif isinstance(value, bool) or isinstance(value, str):
#             grid_params[key] = [value]
#         elif isinstance(value, int) or isinstance(value, float):
#             step = round(abs(value) / (num_samples - 1), 2)
#             values = [
#                 round(value - step * i, 2)
#                 for i in range(num_samples // 2, -num_samples // 2 - 1, -1)
#             ]
#             # ensure all remains as int if they started off that way
#             if isinstance(value, int):
#                 values = [int(val) for val in values]
#             grid_params[key] = values
#     return grid_params


def generate_gridsearch_parameter_grid(params_dict: dict, num_samples: int = 2) -> dict:
    grid_params = {}
    for key, value in params_dict.items():
        if key == "verbose":
            grid_params[key] = [value]
        elif isinstance(value, bool) or isinstance(value, str):
            grid_params[key] = [value]
        elif isinstance(value, int) or isinstance(value, float):
            if num_samples == 0:
                raise ValueError("num_samples cannot be zero")
            elif num_samples % 2 == 0:
                # For even num_samples, space the values around the original value
                step = (
                    round(abs(value) / (num_samples - 1), 2) if num_samples != 1 else 0
                )
                values = [
                    round(value - step * i, 2)
                    for i in range(num_samples // 2, -num_samples // 2 - 1, -1)
                ]
                # Ensure all remains as int if they started off that way
                if isinstance(value, int):
                    values = [int(val) for val in values]
                grid_params[key] = values
            else:
                # For odd num_samples, include the original value and have equal steps around it
                step = round(abs(value) / (num_samples // 2), 2)
                values = [
                    round(value - step * i, 2)
                    for i in range(num_samples // 2, -num_samples // 2 - 1, -1)
                ]
                # Ensure all remains as int if they started off that way
                if isinstance(value, int):
                    values = [int(val) for val in values]
                grid_params[key] = values
    return grid_params


# def initialise_sklearn_random_search(
#     model_instance, cv: int = 3, n_iter: int = 100, n_jobs: int = 4, verbose: int = 0
# ):
#     # model_class = ModelInitializer()
#     # model = model_class.get_model(model_type)

#     search_grid = model_instance.get_random_search_grid()
#     model = model_instance.get_model()

#     return RandomizedSearchCV(
#         estimator=model,
#         param_distributions=search_grid,
#         n_iter=n_iter,
#         cv=cv,
#         verbose=verbose,
#         random_state=model_instance.random_state,
#         n_jobs=n_jobs,
#     )


# def initialise_sklearn_grid_search(
#     model,
#     best_params_dict: dict,
#     cv: int = 3,
#     n_jobs: int = 4,
#     num_samples: int = 3,
#     verbose=0,
# ):
#     param_grid = baselines.generate_parameter_grid(best_params_dict)

#     return GridSearchCV(
#         estimator=model, param_grid=param_grid, cv=cv, n_jobs=n_jobs, verbose=verbose
#     )


def fractional_cover_from_reef_check(
    reef_check_df: pd.DataFrame, substrate_codes: list[str] = ["HC", "SC"]
) -> pd.DataFrame:
    """Calculate coral cover from reef check data
    TODO: calculate for each transect section (four per survey)"""
    # number of observations along whole transect
    NUM_SLOTS = 160

    # group by substrate code and survey_id, summing values in "total" column, and
    # keeping all others the same
    reef_check_df_grouped = reef_check_df.groupby(["survey_id", "substrate_code"]).agg(
        {
            "total": "sum",
            **{
                col: "first"
                for col in reef_check_df.columns
                if col not in ["survey_id", "substrate_code", "total"]
            },
        }
    )

    # specify which rows correspond to substrate of interest
    reef_check_df_grouped["is_substrate"] = (
        reef_check_df_grouped.index.get_level_values(1).isin(substrate_codes)
    )
    # # calculate fractional cover
    reef_check_df_grouped["fractional_cover"] = (
        reef_check_df_grouped["total"] / NUM_SLOTS
    )
    # Filter rows based on the "is_substrate" column
    fractional_cover_df = reef_check_df_grouped[reef_check_df_grouped["is_substrate"]]

    # Define the aggregation dictionary
    agg_dict = {
        "fractional_cover": "sum",
        **{
            col: "first"
            for col in reef_check_df_grouped.columns
            if col not in ["fractional_cover", "is_substrate"]
        },
    }

    # Aggregate by "survey_id" using the defined dictionary
    return fractional_cover_df.groupby("survey_id").agg(agg_dict)


class FileName:
    def __init__(
        self,
        # source_id: str,
        # member_id: str,
        variable_id: str | list,
        grid_type: str,
        fname_type: str,
        date_range: str = None,
        lats: list[float, float] = None,
        lons: list[float, float] = None,
        levs: list[int, int] = None,
        plevels: list[float, float] = None,
    ):
        """
        Args:
            source_id (str): name of climate model
            member_id (str): model run
            variable_id (str): variable name
            grid_type (str): type of grid (tripolar or latlon)
            fname_type (str): type of file (individual, time-concatted, var_concatted) TODO
            lats (list[float, float], optional): latitude range. Defaults to None.
            lons (list[float, float], optional): longitude range. Defaults to None.
            plevels (list[float, float], optional): pressure level range. Defaults to None.
            fname (str, optional): filename. Defaults to None to allow construction.
        """
        self.variable_id = variable_id
        self.grid_type = grid_type
        self.fname_type = fname_type
        self.date_range = date_range
        self.lats = lats
        self.lons = lons
        self.levs = levs
        self.plevels = plevels

    def get_spatial(self):
        if self.lats and self.lons:  # if spatial range specified (i.e. cropping)
            # cast self.lats and self.lons lists to integers. A little crude, but avoids decimals in filenames
            lats = [int(lat) for lat in self.lats]
            lons = [int(lon) for lon in self.lons]
            return lat_lon_string_from_tuples(lats, lons).upper()
        else:
            return "uncropped"

    def get_plevels(self):
        if self.plevels == [-1] or self.plevels == -1:  # seafloor
            return f"sfl-{max(self.levs)}"
        elif not self.plevels:
            return "sfc"
            if self.plevels[0] is None:
                return "sfc"
        elif isinstance(self.plevels, float):  # specified pressure level
            return "{:.0f}".format(self.plevels / 100)
        elif isinstance(self.plevels, list):  # pressure level range
            if self.plevels[0] is None:  # if plevels is list of None, surface
                return "sfc"
            return f"levs_{min(self.plevels)}-{max(self.plevels)}"
        else:
            raise ValueError(
                f"plevel must be one of [-1, float, list]. Instead received '{self.plevels}'"
            )

    def get_var_str(self):
        if isinstance(self.variable_id, list):
            return "_".join(self.variable_id)
        else:
            return self.variable_id

    def get_grid_type(self):
        if self.grid_type == "tripolar":
            return "tp"
        elif self.grid_type == "latlon":
            return "ll"
        else:
            raise ValueError(
                f"grid_type must be 'tripolar' or 'latlon'. Instead received '{self.grid_type}'"
            )

    def get_date_range(self):
        if not self.date_range:
            return None
        if self.fname_type == "time_concatted" or self.fname_type == "var_concatted":
            # Can't figure out how it's being done currently
            return str("-".join((str(self.date_range[0]), str(self.date_range[1]))))
        else:
            return self.date_range

    def join_as_necessary(self):
        var_str = self.get_var_str()
        spatial = self.get_spatial()
        plevels = self.get_plevels()
        grid_type = self.get_grid_type()
        date_range = self.get_date_range()

        # join these variables separated by '_', so long as the variable is not None
        return "_".join(
            [i for i in [var_str, spatial, plevels, grid_type, date_range] if i]
        )

    def construct_fname(self):

        if self.fname_type == "var_concatted":
            if not isinstance(self.variable_id, list):
                raise TypeError(
                    f"Concatted variable requires multiple variable_ids. Instead received '{self.variable_id}'"
                )

        self.fname = self.join_as_necessary()

        return f"{self.fname}.nc"


def process_xa_d(
    xa_d: xa.Dataset | xa.DataArray,
    rename_lat_lon_grids: bool = False,
    rename_mapping: dict = {
        "lat": "latitude",
        "lon": "longitude",
        "y": "latitude",
        "x": "longitude",
        "i": "longitude",
        "j": "latitude",
        "lev": "depth",
    },
    squeeze_coords: str | list[str] = None,
    # chunk_dict: dict = {"latitude": 100, "longitude": 100, "time": 100},
    crs: str = "EPSG:4326",
):
    """
    Process the input xarray Dataset or DataArray by standardizing coordinate names, squeezing dimensions,
    chunking along specified dimensions, and sorting coordinates.

    Parameters
    ----------
        xa_d (xa.Dataset or xa.DataArray): The xarray Dataset or DataArray to be processed.
        rename_mapping (dict, optional): A dictionary specifying the mapping for coordinate renaming.
            The keys are the existing coordinate names, and the values are the desired names.
            Defaults to a mapping that standardizes common coordinate names.
        squeeze_coords (str or list of str, optional): The coordinates to squeeze by removing size-1 dimensions.
                                                      Defaults to ['band'].
        chunk_dict (dict, optional): A dictionary specifying the chunk size for each dimension.
                                     The keys are the dimension names, and the values are the desired chunk sizes.
                                     Defaults to {'latitude': 100, 'longitude': 100, 'time': 100}.

    Returns
    -------
        xa.Dataset or xa.DataArray: The processed xarray Dataset or DataArray.

    """
    temp_xa_d = xa_d.copy()

    if rename_lat_lon_grids:
        temp_xa_d = temp_xa_d.rename(
            {"latitude": "latitude_grid", "longitude": "longitude_grid"}
        )

    for coord, new_coord in rename_mapping.items():
        if new_coord not in temp_xa_d.coords and coord in temp_xa_d.coords:
            temp_xa_d = temp_xa_d.rename({coord: new_coord})
    # temp_xa_d = xa_d.rename(
    #     {coord: rename_mapping.get(coord, coord) for coord in xa_d.coords}
    # )
    if "band" in temp_xa_d.dims:
        temp_xa_d = temp_xa_d.squeeze("band")
    if squeeze_coords:
        temp_xa_d = temp_xa_d.squeeze(squeeze_coords)

    if "time" in temp_xa_d.dims:
        temp_xa_d = temp_xa_d.transpose("time", "latitude", "longitude", ...)
    else:
        temp_xa_d = temp_xa_d.transpose("latitude", "longitude")

    if "grid_mapping" in temp_xa_d.attrs:
        del temp_xa_d.attrs["grid_mapping"]
    # add crs
    #     temp_xa_d.rio.write_crs(crs, inplace=True)
    # if chunk_dict is not None:
    #     temp_xa_d = chunk_as_necessary(temp_xa_d, chunk_dict)
    # sort coords by ascending values
    return temp_xa_d.sortby(list(temp_xa_d.dims))

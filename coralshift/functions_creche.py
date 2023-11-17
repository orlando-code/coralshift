import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features
import xarray as xa
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from pyinterp.backends import xarray
from pyinterp import fill


def rasterize_geodf(
    geo_df: gpd.geodataframe, resolution_lat: float = 1.0, resolution_lon: float = 1.0
) -> np.ndarray:
    """Rasterize a geodataframe to a numpy array.

    Args:
        geo_df (gpd.geodataframe): Geodataframe to rasterize
        resolution_lat (float): Resolution of the raster in degrees latitude
        resolution_lon (float): Resolution of the raster in degrees longitude

    Returns:
        np.ndarray: Rasterized numpy array
    TODO: add crs customisation. Probably from class object elsewhere. Currently assumes EPSG:4326.
    """

    xmin, ymin, xmax, ymax, width, height = lat_lon_vals_from_geo_df(
        geo_df, resolution_lon, resolution_lat
    )
    # Create the transform based on the extent and resolution
    transform = rasterio.transform.from_bounds(xmin, ymin, xmax, ymax, width, height)
    transform.crs = rasterio.crs.CRS.from_epsg(4326)

    # Any chance of a loading bar? No: would have to dig into the function istelf.
    # could be interesting...
    return features.rasterize(
        [(shape, 1) for shape in geo_df["geometry"]],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        all_touched=True,
        dtype=rasterio.uint8,
    )


def raster_to_xarray(
    raster: np.ndarray,
    x_y_limits: np.ndarray,
    resolution_lat: float = 1.0,
    resolution_lon: float = 1.0,
    name: str = "raster_cast_to_xarray",
) -> xa.DataArray:
    """Convert a raster to an xarray DataArray.

    Args:
        raster (np.ndarray): Raster to convert
        resolution_lat (float): Resolution of the raster in degrees latitude
        resolution_lon (float): Resolution of the raster in degrees longitude

    Returns:
        xa.DataArray: DataArray of the raster
    TODO: add attributes kwarg
    """

    lon_min, lat_min, lon_max, lat_max = x_y_limits
    width = int((lon_max - lon_min) / resolution_lon)
    height = int((lat_max - lat_min) / resolution_lat)

    # Create longitude and latitude arrays
    longitudes = np.linspace(lon_min, lon_max, width)
    # reversed because raster inverted
    latitudes = np.linspace(lat_max, lat_min, height)

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
    return xa_array


def lat_lon_vals_from_geo_df(
    geo_df: gpd.geodataframe, resolution_lon: float = 1.0, resolution_lat: float = 1.0
):
    # Calculate the extent in degrees from bounds of geometry objects
    lon_min, lat_min, lon_max, lat_max = geo_df["geometry"].total_bounds
    # Calculate the width and height of the raster in pixels based on the extent and resolution
    width = int((lon_max - lon_min) / resolution_lon)
    height = int((lat_max - lat_min) / resolution_lat)

    return lon_min, lat_min, lon_max, lat_max, width, height


def rasterise_points_df(
    df: pd.DataFrame,
    lat_column: str,
    lon_column: str,
    resolution_lat: float = 1.0,
    resolution_lon: float = 1.0,
    bbox: list[float] = [-90, -180, 90, 180],
) -> np.ndarray:
    """Rasterize a pandas dataframe of points to a numpy array.

    Args:
        df (pd.DataFrame): Dataframe of points to rasterize
        resolution_lat (float): Resolution of the raster in degrees latitude
        resolution_lon (float): Resolution of the raster in degrees longitude

    Returns:
        np.ndarray: Rasterized numpy array"""

    # extract bbox limits of your raster
    min_lat, min_lon, max_lat, max_lon = bbox

    # Calculate the number of rows and columns in the raster
    num_rows = int((max_lat - min_lat) / resolution_lat)
    num_cols = int((max_lon - min_lon) / resolution_lon)

    # Initialize an empty raster (of zeros)
    raster = np.zeros((num_rows, num_cols), dtype=int)

    # Convert latitude and longitude points to row and column indices of raster
    row_indices = ((max_lat - df[lat_column]) // resolution_lat).astype(int)
    col_indices = ((df[lon_column] - min_lon) // resolution_lon).astype(int)

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


def check_var_has_coords(xa_array, coords: list[str] = ["time", "lat", "lon"]):
    """
    Select and return data variables that have 'time', 'latitude', and 'longitude' coordinates.

    Args:
        dataset (xarray.Dataset): Input xarray dataset.

    Returns:
        xarray.Dataset: Subset of the input dataset containing selected data variables.

    N.B. must have consistent naming of coordinate dimensions by this point.
    """
    # Define the coordinate names to check for
    required_coords = ["time", "latitude", "longitude"]

    if all(coord in xa_array.coords for coord in required_coords):
        return True
    else:
        return False


def apply_fill_loess(dataset, nx=2, ny=2):
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

    print(f"{len(buffered_data_vars)} variables to process...")
    for _, (var_name, var_data) in enumerate(buffered_data_vars.items()):
        if check_var_has_coords(var_data):
            for t in tqdm(
                buffered_dataset.time,
                desc=f"Processing timesteps of variable '{var_name}'",
            ):
                grid = xarray.Grid2D(var_data.sel(time=t))
                filled = fill.loess(grid, nx=nx, ny=ny)
                buffered_data_vars[var_name].loc[dict(time=t)] = filled.T

    return buffered_dataset


def resample_xa_d(
    xa_d_list: xa.DataArray | xa.Dataset,
    lat_range: list[float],
    lon_range: list[float],
    resolution_lat: float,
    resolution_lon: float,
    resample_method: str = "linear",
):
    """
    Resample an xarray DataArray or Dataset to a common extent and resolution.

    Args:
        xa_d_list (xa.DataArray | xa.Dataset): xarray DataArray or Dataset to resample.
        lat_range (list[float]): Latitude range of the common extent.
        lon_range (list[float]): Longitude range of the common extent.
        resolution_lat (float): Latitude resolution of the common extent.
        resolution_lon (float): Longitude resolution of the common extent.
        resample_method (str, optional): Resampling method to use. Defaults to "linear".

    Returns:
        xa.DataArray | xa.Dataset: Resampled xarray DataArray or Dataset.
    """
    # Create a dummy dataset with the common extent and resolution
    common_dataset = xa.Dataset(
        coords={
            "latitude": (["latitude"], np.arange(*lat_range, resolution_lat)),
            "longitude": (["longitude"], np.arange(*lon_range, resolution_lon)),
        }
    )

    # Resample the input dataset to the common resolution and extent using "method" interpolation
    resampled_dataset = input_ds.interp(
        latitude=common_dataset["latitude"],
        longitude=common_dataset["longitude"],
        method=resample_method,
    )

    return resampled_dataset


def spatially_combine_xa_d_list(
    xa_d_list: list[xa.DataArray | xa.Dataset],
    lat_range: list[float],
    lon_range: list[float],
    resolution_lat: float,
    resolution_lon: float,
    resample_method: str = "linear",
) -> xa.Dataset:
    """
    Resample and merge a list of xarray DataArrays or Datasets to a common extent and resolution.

    Args:
        xa_d_list (list[xa.DataArray | xa.Dataset]): List of xarray DataArrays or Datasets to resample and merge.
        lat_range (list[float]): Latitude range of the common extent.
        lon_range (list[float]): Longitude range of the common extent.
        resolution_lat (float): Latitude resolution of the common extent.
        resolution_lon (float): Longitude resolution of the common extent.
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
            "latitude": (["latitude"], np.arange(*lat_range, resolution_lat)),
            "longitude": (["longitude"], np.arange(*lon_range, resolution_lon)),
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


def tvt_spatial_split(
    df: pd.DataFrame, tvt_fractions: list[float], orientation: str = "vertical"
) -> list[pd.DataFrame]:
    """
    Splits a dataframe into train, test, and validation sets, spatially.

    Args:
        df (pd.DataFrame): The dataframe to split.
        tvt_fractions (list[float]): The fractions of the dataframe to allocate to train, test, and validation sets.
        orientation (str, optional): The orientation of the splits. Defaults to "vertical".

    Returns:
        list[pd.DataFrame]: The train, test, and validation sets.
    """
    # check that tvt_fractions sum to 1 or if any is zero
    assert (
        sum(tvt_fractions) == 1 or 0 in tvt_fractions
    ), f"tvt_fractions must sum to 1 or contain a zero. Currently sum to {sum(tvt_fractions)}"

    assert orientation in [
        "vertical",
        "horizontal",
    ], f"orientation must be 'vertical' or 'horizontal'. Currently {orientation}"

    if orientation == "horizontal":
        df = df.sort_values(by="latitude")
    elif orientation == "vertical":
        df = df.sort_values(by="longitude")

    df_list = []

    if 0 in tvt_fractions:
        nonzero_fractions = [frac for frac in tvt_fractions if frac != 0]
        split_indices = [int(frac * len(df)) for frac in np.cumsum(nonzero_fractions)]

        for idx, split_idx in enumerate(split_indices):
            if idx == 0:
                df_list.append(df.iloc[:split_idx])
            elif idx == len(split_indices) - 1:
                df_list.append(df.iloc[split_idx:])
            else:
                df_list.append(df.iloc[split_indices[idx - 1] : split_idx])
    else:
        df_list = np.split(
            df,
            [
                int(tvt_fractions[0] * len(df)),
                int((tvt_fractions[0] + tvt_fractions[1]) * len(df)),
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
        df_tvt_list = tvt_spatial_split(df, train_val_test_frac)
    elif split_type == "pixel":
        print("To implement")

    # generate and save scaling parameters, ignoring nans. Start with min-max scaling
    scaling_params_dict = scaling_params_from_df(df_tvt_list[0], predictors + [gt])
    # scale train, val, test according to train scaling parameters
    df_tvt_scaled_list = [
        scale_dataframe(df, scaling_params_dict) for df in df_tvt_list
    ]
    # one-hot encode nans
    df_tvt_scaled_onehot_list = [
        onehot_df(scaled_df) for scaled_df in df_tvt_scaled_list
    ]
    if 0 in train_val_test_frac:
        # cast to numpy arrays
        trains, tests = [
            Xs_ys_from_df(df, predictors + ["onehot_nan"], gt)
            for df in df_tvt_scaled_onehot_list
        ]
        return (trains, (0, 0), tests), df_tvt_scaled_list
    else:
        # cast to numpy arrays
        trains, vals, tests = [
            Xs_ys_from_df(df, predictors + ["onehot_nan"], gt)
            for df in df_tvt_scaled_onehot_list
        ]
        return (trains, vals, tests), df_tvt_scaled_list


def reform_df(df: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    """
    Reformats a DataFrame to include a column of predictions.
    """
    assert len(df) == len(
        predictions
    ), "Length of DataFrame and predictions array must be equal."

    # Create a copy of the original DataFrame to avoid modifying it in place
    df_predictions = df.copy()
    # add prediction column to df
    df_predictions["prediction"] = predictions

    return df_predictions

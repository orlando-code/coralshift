import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features
from datetime import datetime
import calendar
import xarray as xa
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from pyinterp.backends import xarray
from pyinterp import fill

from coralshift.processing import spatial_data


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
        name=name
    )
    # Set the CRS (coordinate reference system) if needed
    # TODO: make kwarg
    xa_array.attrs["crs"] = "EPSG:4326"  # Example CRS, use the appropriate CRS
    # TODO: set attributes if required
    #     attrs=dict(
    #         description="Rasterised Reef Check coral presence survey data"
    #     ))
    return spatial_data.process_xa_d(xa_array)


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
    xa_d: xa.DataArray | xa.Dataset,
    lat_range: list[float],
    lon_range: list[float],
    resolution_lat: float,
    resolution_lon: float,
    resample_method: str = "linear",
):
    """
    Resample an xarray DataArray or Dataset to a common extent and resolution.

    Args:
        xa_d (xa.DataArray | xa.Dataset): xarray DataArray or Dataset to resample.
        lat_range (list[float]): Latitude range of the common extent.
        lon_range (list[float]): Longitude range of the common extent.
        resolution_lat (float): Latitude resolution of the common extent.
        resolution_lon (float): Longitude resolution of the common extent.
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
    # Create a dummy dataset with the common extent and resolution
    common_dataset = xa.Dataset(
        coords={
            "latitude": (["latitude"], np.arange(*lat_range, resolution_lat)),
            "longitude": (["longitude"], np.arange(*lon_range, resolution_lon)),
        }
    )

    # Resample the input dataset to the common resolution and extent using "method" interpolation
    resampled_dataset = xa_d.interp(
        latitude=common_dataset["latitude"],
        longitude=common_dataset["longitude"],
        method=resample_method,
    )

    return resampled_dataset


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
    # Resample the input dataset to the resolution and extent of the target dataset
    resampled_dataset = xa_d_to_resample.interp(
        latitude=target_xa["latitude"],
        longitude=target_xa["longitude"],
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
            "latitude": (["latitude"], np.arange(*np.sort(lat_range), resolution_lat)),
            "longitude": (["longitude"], np.arange(*np.sort(lon_range), resolution_lon)),
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

    # this was hanlding omission of val completely: for now, keeping it in, but just empty
    # if 0 in tvt_fractions:
    #     nonzero_fractions = [frac for frac in tvt_fractions if frac != 0]
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
    df_tvt_list = None # trying to free up memory
    # one-hot encode nans
    df_tvt_scaled_onehot_list = [
        onehot_df(scaled_df) for scaled_df in df_tvt_scaled_list
    ]
    df_tvt_scaled_list = None # trying to free up memory

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


def year_to_datetime(year: int, xa_d: xa.DataArray | xa.Dataset = None) -> datetime:
    """
    Convert an integer denoting a year to a datetime object.

    Args:
        year (int): The year to convert.
        xa_d (xa.DataArray | xa.Dataset, optional): An xarray DataArray or Dataset to check the year against. Defaults to None.

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

    if not vars:
        vars = dataset_period.data_vars.keys()

    for i, var_name in tqdm(
        enumerate(vars), desc=f"calculating statistics for variables", total=len(vars)
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
        time_std = var_data.std(dim="time")
        stats[f"{var_name}_time_std"] = time_std.values

        # Calculate standard deviation of January and July values
        january_std = var_data.where(var_data["time.month"] == 1).std(dim="time")
        july_std = var_data.where(var_data["time.month"] == 7).std(dim="time")
        stats[f"{var_name}_jan_std"] = january_std.values
        stats[f"{var_name}_jul_std"] = july_std.values

        # Calculate the overall mean for each statistic
        stats[f"{var_name}_overall_mean"] = var_data.mean(dim="time").values

    # Combine all calculated variables into a new dataset
    stats_xa = xa.Dataset(
        {key: (("latitude", "longitude"), value) for key, value in stats.items()},
        coords={"latitude": var_data.latitude, "longitude": var_data.longitude},
    )

    return stats_xa


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


def tuples_to_string(lats, lons):
    # Round the values in the tuples to the nearest integers
    round_lats = [round(lat) for lat in lats]
    round_lons = [round(lon) for lon in lons]

    # Create the joined string
    return f"lats_{min(round_lats)}-{max(round_lats)}_lons_{min(round_lons)}-{max(round_lons)}"


def split_dataset_and_save(ds_fp, divisor, output_dir_name: str=None, select_vars: list[str]=None):
    
    ds = xa.open_dataset(ds_fp)
    
    if select_vars:
        ds = ds[select_vars]
            
    subsets_dict = split_dataset_by_indices(ds, divisor)
    
    # Create a subdirectory to save the split datasets
    if output_dir_name:
        output_dir = Path(ds_fp).parent / f"{output_dir_name}_{divisor**2}_split_datasets"
    else:
        output_dir = Path(ds_fp).parent / f"{divisor**2}_split_datasets"
    output_dir.mkdir(parents=True, exist_ok=True)

    
    for coord_info, subset in tqdm(subsets_dict.items(), desc="saving dataset subsets..."):
        stem_stem = str(Path(ds_fp).stem).split("lats")[0]
        # Construct the filename based on bounds
        filename = f"{stem_stem}_{coord_info}.nc"
        save_fp = output_dir / filename
        subset.to_netcdf(save_fp)
    return subsets_dict
    

def split_dataset_by_indices(dataset, divisor) -> dict:
    subsets_dict = {}
    num_lats = len(dataset.latitude.values) // divisor
    num_lons = len(dataset.longitude.values) // divisor
    for i in range(divisor):
        for j in range(divisor):
            start_lat_ind = i * num_lats
            start_lon_ind = j * num_lons
            
            subset = dataset.isel(latitude=slice(start_lat_ind, start_lat_ind + num_lats),
                                  longitude=slice(start_lon_ind, start_lon_ind + num_lons))
            
            lat_lims = spatial_data.min_max_of_coords(subset, "latitude")
            lon_lims = spatial_data.min_max_of_coords(subset, "longitude")
            
            coord_info = functions_creche.tuples_to_string(lat_lims, lon_lims)
            subsets_dict[coord_info] = subset
    
    return subsets_dict


def ds_to_ml_ready(ds, 
    gt:str="unep_coral_presence", exclude_list: list[str]=["latitude", "longitude", "latitude_grid", "longitude_grid", "crs", "depth", "spatial_ref"], 
    train_val_test_frac=[1,0,0], inf_type: str="classification", threshold=0.5, depth_mask_lims = [-50, 0], client=None, remove_rows:bool=False):
    
    df = ds.compute().to_dataframe()
    # TODO: implement checking for empty dfs

    predictors = [pred for pred in df.columns if pred != gt and pred not in exclude_list]
    depth_condition = (df["elevation"] < max(depth_mask_lims)) & (df["elevation"] > min(depth_mask_lims))
    
    if remove_rows:
        df = df[depth_condition]
    else:
        df["within_depth"] = 0
        df.loc[depth_condition, "within_depth"] = 1
        
    if len(df) > 0:
        scaler = MinMaxScaler()
        df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index = df.index)
    
    df["nan_onehot"] = df.isna().any(axis=1).astype(int)
    df = df.fillna(0)
    
#     X = df[predictors].to_numpy()
#     y = df[gt].to_numpy()
    
    X = df[predictors]
    y = df[gt]
    
    return X, y

def cont_to_class(array, threshold=0.5):
    array[array >= threshold] = 1
    array[array < threshold] = 0

    return array.astype(int)


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

    return fig, ax
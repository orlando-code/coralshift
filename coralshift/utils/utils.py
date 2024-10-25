# general
import numpy as np
import pandas as pd
from datetime import datetime
import os

# spatial
import geopandas as gpd

# spatial
import xarray as xa

# custom
from coralshift.processing import spatial_data


def flatten_dict(dictionary: dict, parent_key: str = "", sep: str = "_"):
    """
    Recursively flattens a nested dictionary into a single-level dictionary.

    Args:
        dictionary (dict): The nested dictionary to be flattened.
        parent_key (str, optional): The parent key to be used for the flattened keys. Defaults to an empty string.
        sep (str, optional): The separator to be used between the parent key and the child key. Defaults to '_'.

    Returns:
        dict: The flattened dictionary.
    """
    items = []
    for k, v in dictionary.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def is_type_or_list_of_type(obj, target_type) -> bool:
    """Checks if an object or a list/tuple of objects is of a specific type.

    Parameters
    ----------
        obj: Object or list/tuple of objects to be checked.
        target_type: Target type to check against.

    Returns
    -------
        bool: True if the object or list/tuple of objects is of the specified type, False otherwise.
    """
    if isinstance(obj, target_type):
        return True

    if isinstance(obj, list) or isinstance(obj, tuple):
        return all(isinstance(element, target_type) for element in obj)

    return False


def cast_to_list(obj):
    if not is_type_or_list_of_type(obj, list):
        return [obj]
    else:
        return obj


def flatten_list(nested_list: list[list]) -> list:
    if len(nested_list) == 1:
        return nested_list[0]
    else:
        return [element for sublist in nested_list for element in sublist]


def remove_duplicates_from_dict(dictionaries):
    return [dict(t) for t in {tuple(sorted(d.items())) for d in dictionaries}]


def get_multiindex_min_max(dataframe):
    min_max_dict = {}

    for level_name in dataframe.index.names:
        level_values = dataframe.index.get_level_values(level_name)
        min_value = level_values.min()
        max_value = level_values.max()
        min_max_dict[level_name] = {"min": min_value, "max": max_value}

    return min_max_dict


def round_list_tuples(
    tuple_list: list[tuple[float, ...]], decimal_places: int = 2
) -> list[tuple]:
    """Round each element in a tuple to a specified precision.

    Parameters
    ----------
        tup (tuple[float]): A tuple of floats.
        prec (int): The precision to round the elements to. Defaults to 2.

    Returns
    -------
        list[str]: A list of rounded values as strings.
    """
    if tuple_list.isinstance(tuple):
        tuple_list = [tuple_list]
    return [
        tuple(round(element, decimal_places) for element in sub_tuple)
        for sub_tuple in tuple_list
    ]


def underscore_str_of_dates(dts: list[str | np.datetime64]) -> list[str]:
    """Extract date strings from a list of datetime objects.

    Parameters
    ----------
        dts (list[str | np.datetime64]): A list of datetime objects.

    Returns
    -------
        list[str]: A list of date strings.
    """
    if dts.isinstance(list) or dts.isinstance(tuple):
        return "_".join([spatial_data.date_from_dt(dt) for dt in dts])
    else:
        return spatial_data.date_from_dt(dts)


def underscore_str_of_strings(variables: str | list[str]) -> str:
    """Convert variable(s) to a string.

    Parameters
    ----------
        variables (str | list[str]): A single variable as a string or a list of variables.

    Returns
    -------
        str: A string representation of the variable(s).
    """
    # if single variable
    if variables.isinstance(str):
        return variables
    else:
        return "_".join([str(var) for var in variables])


def underscore_list_of_tuples(tuples: str | list[tuple]) -> str:
    """Converts a list of tuples or a single tuple into a string with elements separated by underscores.

    Parameters
    ----------
        tuples (str or list[tuple]): List of tuples or a single tuple.

    Returns
    -------
        str: String representation of the tuples with elements separated by underscores.
    """
    if tuples.isinstance(list):
        flattened_list = [item for sublist in tuples for item in sublist]
        return "_".join(map(str, flattened_list))
    else:
        return "_".join([str(tup) for tup in tuples])


def generate_date_pairs(
    date_lims: tuple[str, str], freq: str = "1y"
) -> list[tuple[str, str]]:
    """
    Generate pairs of start and end dates based on the given date limits.

    Parameters
    ----------
        date_lims (tuple[str, str]): A tuple containing the start and end dates.
        freq (str): frequency with which to sample times

    Returns:
    date_pairs (list[tuple[str, str]]): A list of date pairs.
    """

    start_overall, end_overall = pd.to_datetime(min(date_lims)), pd.to_datetime(
        max(date_lims)
    )

    date_list = pd.date_range(date_lims[0], date_lims[1], freq=freq)

    if len(date_list) < 1:
        date_list = [np.datetime64(start_overall), np.datetime64(end_overall)]
    return [
        (
            np.datetime64(date_list[i]),
            np.datetime64(date_list[i + 1]),
        )
        for i in range(len(date_list) - 1)
    ]


def get_buffered_lims(coord_vals, buffer_size=1):   # for security
    return [min(coord_vals) - buffer_size, max(coord_vals) + buffer_size]


def replace_dot_with_dash(string: str) -> str:
    """
    Replace all occurrences of "." with "-" in a string.

    Parameters
    ----------
        string (str): The input string.

    Returns
    -------
        str: The modified string with "." replaced by "-".
    """
    return string.replace(".", "-")


def pad_number_with_zeros(number: str | int, resulting_len: int = 2) -> str:
    """Add leading zeros to a number until the desired length. Useful for generating dates in URL strings or any other
    scenario where leading zeros are required.

    Parameters
    ----------
    number (str | int): The number to be padded with zeros.
    resulting_len (int, optional): The desired length of the resulting string (default is 2).

    Returns
    -------
    str: The padded number as a string.
    """
    if not isinstance(number, str):
        try:
            number = str(number)
        except ValueError:
            print(f"Failed to convert {number} to string")

    while len(number) < resulting_len:
        number = "".join(("0", number))

    return number


def mask_above_threshold(xa_d, threshold=10000):
    """
    Mask values above a given threshold with np.nan in an xarray Dataset or DataArray.

    Parameters:
    xr_obj (xarray.Dataset or xarray.DataArray): The input xarray object.
    threshold (float): The threshold value above which the data will be masked with np.nan. Default is 5000.

    Returns:
    xarray.Dataset or xarray.DataArray: The masked xarray object.
    """
    if isinstance(xa_d, xa.Dataset):
        # print(type(xa_d))
        # print(np.min(xa_d.lat.values))
        # TODO: not all variable types will work with this
        return xa_d.map(lambda x: x.where(x <= threshold, np.nan))
    elif isinstance(xa_d, xa.DataArray):
        return xa_d.where(xa_d <= threshold, np.nan)
    else:
        raise TypeError("Input must be an xarray Dataset or DataArray")


def select_df_rows_by_coords(df: pd.DataFrame, coordinates: list) -> pd.DataFrame:
    """
    Select rows from a Pandas DataFrame based on matching latitude and longitude values.

    Parameters
    ----------
        dataframe (pd.DataFrame): The Pandas DataFrame.
        coordinates (list): List of tuples containing latitude and longitude values.

    Returns
    -------
        pd.DataFrame: The selected subset of the DataFrame.
    """

    # cast to list of tuples in order to write into a set
    coord_set = set(tuple(coord) for coord in coordinates)
    matching_rows = df.index.isin(coord_set)

    return df.iloc[matching_rows]


def list_if_not_already(anything):
    return [item if isinstance(anything, list) else [item] for item in anything]


def drop_nan_rows(
    df: pd.DataFrame, ignore_columns: list[str] = ["latitude", "longitude"]
):
    # Get the columns to consider for dropping rows
    columns_to_check = df.columns.difference(ignore_columns)

    # Drop rows where all columns (excluding the specified ones) contain np.nan
    df_dropped = df.dropna(subset=columns_to_check, how="all")

    return df_dropped


def generate_resolution_str(resolution_d: float = 1 / 27, sfs: int = 4) -> str:
    # TODO: allow specification of sfs
    return replace_dot_with_dash(f"{resolution_d:.04f}d")


def check_discrete(array):
    # Check if all elements are whole numbers (integers)
    if np.all(np.equal(np.mod(array, 1), 0)):
        return True
    # Check if any elements are decimals (floating-point numbers)
    elif np.any(np.not_equal(np.mod(array, 1), 0)):
        return False
    else:
        return False


def convert_to_numeric(value):
    try:
        return pd.to_numeric(value)
    except (ValueError, TypeError):
        return np.nan


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


def get_min_max_coords(ds: xa.Dataset, coord_name: str):
    """
    Get the minimum and maximum values of a given coordinate in a Dataset.

    Parameters:
        ds (xa.Dataset): The input Dataset.
        coord_name (str): The name of the coordinate to get the minimum and maximum values for.

    Returns:
        tuple: A tuple containing the minimum and maximum values of the given coordinate.
    """
    min_coord = float(min(ds[coord_name]).values)
    max_coord = float(max(ds[coord_name]).values)
    return min_coord, max_coord


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


def lat_lon_vals_from_geo_df(geo_df: gpd.geodataframe, resolution: float = 1.0):
    # Calculate the extent in degrees from bounds of geometry objects
    lon_min, lat_min, lon_max, lat_max = geo_df["geometry"].total_bounds
    # Calculate the width and height of the raster in pixels based on the extent and resolution
    width = int((lon_max - lon_min) / resolution)
    height = int((lat_max - lat_min) / resolution)

    return lon_min, lat_min, lon_max, lat_max, width, height


def calc_non_zero_ratio(data, predictand=None):
    if isinstance(data, pd.Series):
        return np.where(data > 0, 1, 0).sum() / len(data)
    if isinstance(data, xa.DataArray):
        np.where(data.flatten() > 0, 1, 0).sum() / len(data.flatten())
    else:
        try:
            return np.where(data[predictand] > 0, 1, 0).sum() / len(data)
        except KeyError:
            raise ValueError(f"Predictand {predictand} not found in DataFrame/Dataset")


# def lat_lon_string_from_tuples(
#     lats: tuple[float, float], lons: tuple[float, float], dp: int = 0
# ):
#     round_lats = iterative_to_string_list(lats, dp)
#     round_lons = iterative_to_string_list(lons, dp)

#     return (
#         f"n{max(round_lats)}_s{min(round_lats)}_w{min(round_lons)}_e{max(round_lons)}"
#     )

def lat_lon_string_from_tuples(
    lats: tuple[float, float], lons: tuple[float, float], dp: int = 0
):
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)

    lats = [round_to_1_sf_after_decimal(min_lat), round_to_1_sf_after_decimal(max_lat)]
    lons = [round_to_1_sf_after_decimal(min_lon), round_to_1_sf_after_decimal(max_lon)]

    lats_strs = [f"s{replace_dot_with_dash(str(abs(round(lat, 1))))}" if lat < 0 else f"n{replace_dot_with_dash(str(abs(round(lat, 1))))}" for lat in lats]   # noqa
    lons_strs = [f"w{replace_dot_with_dash(str(abs(round(lon, 1))))}" if lon < 0 else f"e{replace_dot_with_dash(str(abs(round(lon, 1))))}" for lon in lons]   # noqa

    return "_".join(lats_strs + lons_strs)


def round_to_1_sf_after_decimal(num):
    """
    Rounds a number to 1 significant figure after the decimal point.

    Args:
        num (float): The number to be rounded.

    Returns:
        float: The rounded number.

    Examples:
        >>> round_to_1_sf_after_decimal(3.14159)
        3.1
        >>> round_to_1_sf_after_decimal(10.567)
        10.6
        >>> round_to_1_sf_after_decimal(0.005)
        0.0
    """
    parts = str(num).split('.')
    # if there is a decimal part
    if len(parts) == 2:
        dec_part = parts[1]

        # determine number of zeros before first non-zero digit
        num_zeros = 0
        for i in range(len(dec_part)):
            if dec_part[i] == '0':
                num_zeros += 1
            else:
                break
        return float(parts[0] + '.' + num_zeros*'0' + f"{float(f"{float(dec_part):.1g}"):g}"[0])    # noqa
    else:
        return float(parts[0])


def iterative_to_string_list(iter_obj: tuple, dp: int = 0):
    # Round the values in the iterable object to the specified number of decimal places
    return [round(i, dp) for i in iter_obj]


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


def count_nonzero_values(xa_da: xa.DataArray) -> int:
    """
    Count the number of nonzero values in a DataArray.

    Parameters
    ----------
        xa_da (xa.DataArray): DataArray containing values.

    Returns
    -------
        int: Number of nonzero values.
    """
    non_zero_count = xa_da.where(xa_da != 0).count().item()
    return non_zero_count


def num_cpus():
    return os.cpu_count()


def calc_worker_memory_lim(n_workers, memory_limit):
    return memory_limit / n_workers
    

def memory_string(memory_limit):
    return str(memory_limit) + "GB"


# def round_to_1_sf_after_decimal(num):
#     """
#     Rounds a number to 1 significant figure after the decimal point.

#     Args:
#         num (float): The number to be rounded.

#     Returns:
#         float: The rounded number.

#     Examples:
#         >>> round_to_1_sf_after_decimal(3.14159)
#         3.1
#         >>> round_to_1_sf_after_decimal(10.567)
#         10.6
#         >>> round_to_1_sf_after_decimal(0.005)
#         0.0
#     """
#     parts = str(num).split('.')
#     # if there is a decimal part
#     if len(parts) == 2:
#         dec_part = parts[1]

#         # determine number of zeros before first non-zero digit
#         num_zeros = 0
#         for i in range(len(dec_part)):
#             if dec_part[i] == '0':
#                 num_zeros += 1
#             else:
#                 break
#         return float(parts[0] + '.' + num_zeros*'0' + f"{float(f"{float(dec_part):.1g}"):g}"[0])    # noqa
#     else:
#         return float(parts[0])

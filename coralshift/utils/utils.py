from __future__ import annotations

import pandas as pd
import numpy as np
from coralshift.processing import spatial_data


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

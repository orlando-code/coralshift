from __future__ import annotations

import pandas as pd
import numpy as np
from coralshift.processing import data


def is_type_or_list_of_type(obj, target_type):
    if isinstance(obj, target_type):
        return True

    if isinstance(obj, list) or isinstance(obj, tuple):
        return all(isinstance(element, target_type) for element in obj)

    return False


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
    if type(tuple_list) == tuple:
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
    # dates = []
    # for dt in dts:
    #     dates.append(data.date_from_dt(dt))
    # return dates
    if type(dts) == list or type(dts) == tuple:
        return "_".join([data.date_from_dt(dt) for dt in dts])
    else:
        return data.date_from_dt(dts)


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
    if type(variables) == str:
        return variables
    else:
        return "_".join(variables)


def underscore_list_of_tuples(tuples: str | list[tuple]) -> str:
    if type(tuples) == list:
        flattened_list = [item for sublist in tuples for item in sublist]
        return "_".join(map(str, flattened_list))
    else:
        return "_".join([str(tup) for tup in tuples])


def generate_date_pairs(
    date_lims: tuple[str, str], freq: str = "10y"
) -> list[tuple[str, str]]:
    """Generate pairs of start and end dates based on the given date limits.

    Parameters:
    date_lims (tuple[str, str]): A tuple containing the start and end dates.

    Returns:
    date_pairs (list[tuple[str, str]]): A list of date pairs.
    """
    start_overall, end_overall = min(date_lims), max(date_lims)
    if (end_overall - start_overall).item().days <= 2:
        return [date_lims]

    date_list = pd.date_range(date_lims[0], date_lims[1], freq=freq)
    return [
        (
            np.datetime64(date_list[i]),
            np.datetime64(date_list[i + 1]),
        )
        for i in range(len(date_list) - 1)
    ]


def pad_number_with_zeros(number: str | int) -> str:
    """
    Add a leading zero to any number, X, into 0X. Useful for generating dates in URL strings.
    """
    if not type(number) == str:
        try:
            number = str(number)
        except ValueError:
            print(f"Failed to convert {number} to string")
    if len(number) == 1:
        number = "".join(("0", number))

    return number

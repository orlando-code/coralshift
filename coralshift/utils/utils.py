import pandas as pd
import numpy as np
from coralshift.processing import data


def round_list_tuples(tup_list: list) -> list[list[str]]:
    """Round the elements of each tuple in the list and return a new rounded list.

    Parameters
    ----------
    tup_list (list[tuple[float]]): A list of tuples containing float values.

    Returns
    -------
    rounded (list[list[str]]): A new list with rounded elements of the input tuples.
    """

    rounded = []
    for tup in tup_list:
        rounded.append(round_tuple_els(tup))
    return rounded


def round_tuple_els(tup: tuple[float], prec_str: str = "{:.2f}") -> list[str]:
    """Round each element in a tuple to a specified precision.

    Parameters
    ----------
    tup (tuple[float]): A tuple of floats.
    prec (int): The precision to round the elements to. Defaults to 2.

    Returns
    -------
    List[str]: A list of rounded values as strings.
    """
    rounded = []
    for el in tup:
        rounded.append(prec_str.format(el))
    return rounded


def dates_from_dt(dts: list[str | np.datetime64]) -> list[str]:
    """Extract date strings from a list of datetime objects.

    Parameters
    ----------
    dts (list[str | np.datetime64]): A list of datetime objects.

    Returns
    -------
    list[str]: A list of date strings.
    """
    dates = []
    for dt in dts:
        dates.append(data.date_from_dt(dt))
    return dates


def vars_to_strs(variables: str | list[str]) -> str:
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


def generate_date_pairs(
    date_lims: tuple[str, str], freq: str = "W"
) -> list[tuple[str, str]]:
    """Generate pairs of start and end dates based on the given date limits.

    Parameters:
    date_lims (tuple[str, str]): A tuple containing the start and end dates.

    Returns:
    date_pairs (list[tuple[str, str]]): A list of date pairs.
    """
    start_overall, end_overall = min(date_lims), max(date_lims)
    # if already less than a month apart
    if (end_overall - start_overall).item().days <= 30:
        return [date_lims]

    date_list = pd.date_range(date_lims[0], date_lims[1], freq=freq)
    return [(date_list[i], date_list[i + 1]) for i in range(len(date_list) - 1)]

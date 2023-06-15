from __future__ import annotations

import xarray as xa
import numpy as np


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
        offset = "W"

    return group, offset


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
    weighted_av = calc_weighted_mean(xa_da_daily_means, period).rename(
        base_name + "mean"
    )
    # standard deviation of weighted averages
    stdev = (
        weighted_av.std(dim="time", skipna=True)
        .rename(base_name + "std")
        .rio.write_crs(crs, inplace=True)
    )
    # max and min
    min = (
        xa_da_daily_means.min(dim="time", skipna=True)
        .rename(base_name + "min")
        .rio.write_crs(crs, inplace=True)
    )
    max = (
        xa_da_daily_means.max(dim="time", skipna=True)
        .rename(base_name + "max")
        .rio.write_crs(crs, inplace=True)
    )

    # Return the weighted average
    return weighted_av, stdev, (min, max)


def calculate_magnitude(
    horizontal_data: xa.DataArray, vertical_data: xa.DataArray
) -> xa.DataArray:
    """
    Calculates the resultant magnitude of horizontal and vertical data.

    Parameters
    ----------
    horizontal_data (xarray.DataArray): The input xarray DataArray of horizontal data.
    vertical_data (xarray.DataArray): The input xarray DataArray of vertical data.

    Returns
    -------
    xarray.DataArray: The magnitude of the horizontal and vertical data.
    """
    # determine original crs
    crs_h, crs_v = horizontal_data.rio.crs, vertical_data.rio.crs
    if crs_h != crs_v:
        raise ValueError(
            f"Unmatching crs values in xa.DataArrays: horizontal crs = {crs_h}, vertical crs = {crs_v}"
        )

    def magnitude(horizontal_data, vertical_data) -> xa.DataArray:
        return np.sqrt(horizontal_data**2 + vertical_data**2)

    # func = lambda horizontal_data, vertical_data: np.sqrt(
    #     horizontal_data**2 + vertical_data**2
    # )
    # return xa.apply_ufunc(func, horizontal_data, vertical_data)
    return xa.apply_ufunc(magnitude, horizontal_data, vertical_data).rio.write_crs(
        crs_h, inplace=True
    )

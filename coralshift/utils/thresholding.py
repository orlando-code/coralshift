# general
import numpy as np
import pandas as pd
import xarray as xa
from scipy.interpolate import griddata

# plotting
import matplotlib.pyplot as plt


def calc_stats(da: xa.DataArray, dims: list = ["latitude", "longitude"]) -> dict:
    """
    Compute statistics across specified dimensions (mean, percentiles, min, max)

    Args:
        da (xa.DataArray): DataArray with specified dimensions
        dims (list): List of dimensions to compute statistics across

    Returns:
        dict: Dictionary with mean, 10th, 90th, 25th, 75th, min, and max values
    """
    # rechunk to allow calculation
    da = da.chunk({"time": -1})
    # select first depth if depth present TODO: a bit janky
    if "depth" in da.dims:
        da = da.isel(depth=0)
    stats = {
        "mean": da.mean(dim=dims).values,
        # "median": da.median(dim=dims).values,
        "10th": da.quantile(0.1, dim=dims).values,
        "90th": da.quantile(0.9, dim=dims).values,
        "25th": da.quantile(0.25, dim=dims).values,
        # "50th": da.quantile(0.5, dim=dims).values,
        "75th": da.quantile(0.75, dim=dims).values,
        "min": da.min(dim=dims).values,
        "max": da.max(dim=dims).values,
    }
    return stats


def calc_alltime_stats(da: xa.DataArray) -> dict:
    """
    Compute statistics across spatial and time dimensions (mean, percentiles, min, max)

    Args:
        da (xa.DataArray): DataArray with "latitude", "longitude", and "time" dimensions

    Returns:
        dict: Dictionary with mean, 10th, 90th, 25th, 75th, min, and max values
    """
    return calc_stats(da, ["latitude", "longitude", "time"])


def plot_timeseries_with_quantiles(
    ds: xa.DataArray,
    ylabel: str = None,
    plot_key: str = None,
    fig: plt.Figure = None,
    **kwargs,
) -> plt.Figure:
    """
    Plot a time series with mean, min, max, and 10th, 25th, 75th, and 90th percentiles

    Args:
        ds (xa.DataArray): DataArray with time series values
        ylabel (str): Label for the y-axis
        fig (plt.Figure, optional): Matplotlib figure object to plot on. Defaults to None.
        **kwargs: Additional keyword arguments for customizing the plot

    Returns:
        plt.Figure: Matplotlib figure object
    """
    stats = calc_stats(ds)

    # Set default plotting parameters
    default_params = {
        "figsize": (12, 6),
        "dpi": 100,
        "color_min_max": "grey",
        "ls_min_max": "-.",
        "alpha_min_max": 0.2,
        "color_fill_90th": "blue",
        "alpha_fill_90th": 0.2,
        "color_fill_50th": "blue",
        "alpha_fill_50th": 0.4,
        "color_mean": "blue",
        "linewidth_mean": 2,
    }

    # Update default parameters with any user-provided kwargs
    plot_params = {**default_params, **kwargs}

    if fig is None:
        # Create a new figure if none is provided
        fig, ax = plt.subplots(figsize=plot_params["figsize"], dpi=plot_params["dpi"])
    else:
        # Use the provided figure and get the current axis
        ax = fig.gca()

    # fig.patch.set_visible(False)
    # ax.patch.set_visible(False)

    time = ds["time"].values

    # Plot min and max as faint dotted lines
    ax.plot(
        time,
        stats["min"],
        color=plot_params["color_min_max"],
        linestyle=plot_params["ls_min_max"],
        alpha=plot_params["alpha_min_max"],
    )
    ax.plot(
        time,
        stats["max"],
        color=plot_params["color_min_max"],
        linestyle=plot_params["ls_min_max"],
        alpha=plot_params["alpha_min_max"],
    )
    ax.plot(
        [],
        [],
        color=plot_params["color_min_max"],
        linestyle=plot_params["ls_min_max"],
        label=f"{plot_key} Min/Max" if plot_key else "Min/Max",
    )
    # Fill between 10th and 90th percentiles
    ax.fill_between(
        time,
        stats["10th"],
        stats["90th"],
        color=plot_params["color_fill_90th"],
        alpha=plot_params["alpha_fill_90th"],
        label=f"{plot_key} 90th percentiles" if plot_key else "90th percentiles",
    )

    # Fill between 25th and 75th percentiles
    ax.fill_between(
        time,
        stats["25th"],
        stats["75th"],
        color=plot_params["color_fill_50th"],
        alpha=plot_params["alpha_fill_50th"],
        label=f"{plot_key} 50th percentiles" if plot_key else "50th percentiles",
    )

    # Plot mean as a thick line
    ax.plot(
        time,
        stats["mean"],
        color=plot_params["color_mean"],
        linewidth=plot_params["linewidth_mean"],
        label=f"{plot_key} Mean" if plot_key else "Mean",
    )

    # Customize plot appearance
    ax.set_xlabel("Time", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14) if ylabel else None
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def calc_par(
    bathymetry_da: xa.DataArray, rsdo_da: xa.DataArray, kd490_da: xa.DataArray
) -> xa.Dataset:
    """
    Calculate the photosynthetically available radiation (PAR) at the seafloor

    Args:
        bathymetry_da (xa.DataArray): DataArray with bathymetry values
        rsdo_da (xa.DataArray): DataArray with downwelling shortwave radiation values
        kd490_da (xa.DataArray): DataArray with diffuse attenuation coefficient values

    Returns:
        xa.Dataset: Dataset with PAR values
    """
    return (rsdo_da * np.exp(kd490_da * bathymetry_da)).to_dataset(name="par")


def standardize_time(ds: xa.Dataset | xa.DataArray) -> xa.Dataset | xa.DataArray:
    """
    Standardize the time dimension of a dataset to the first day of the month

    Args:
        ds (xa.Dataset): Dataset with a "time" dimension

    Returns:
        xa.Dataset: Dataset with the "time" dimension standardized to the first day of the month
    """
    try:
        # Set all timestamps to the first day of the month
        ds["time"] = ds.indexes["time"].to_period("M").to_timestamp()
    except AttributeError:
        time_index = pd.DatetimeIndex(ds.indexes["time"])
        monthly_timestamps = time_index.to_period("M").to_timestamp()
        ds = ds.assign_coords(time=monthly_timestamps)
    return ds


# Adjust bilinear interpolation function to use nearest neighbors
def nearest_neighbor_fill(data):
    """
    Fill NaNs in a dataset using nearest neighbor interpolation.

    Parameters:
        data: xarray.DataArray
            The input data with NaNs to be filled.
    Returns:
        xarray.DataArray
            Data with NaNs filled by nearest neighbor interpolation.
    """
    # Extract coordinates and data values
    y, x = data["latitude"], data["longitude"]
    values = data.values

    # Create grid points
    grid_y, grid_x = np.meshgrid(y, x, indexing="ij")

    # Mask valid (non-NaN) and invalid (NaN) points
    valid_points = ~np.isnan(values)
    invalid_points = np.isnan(values)

    # Perform interpolation
    interpolated = griddata(
        (grid_y[valid_points], grid_x[valid_points]),
        values[valid_points],
        (grid_y[invalid_points], grid_x[invalid_points]),
        method="nearest",  # Nearest neighbor interpolation
    )

    # Replace NaNs in original data with interpolated values
    filled_values = values.copy()
    filled_values[invalid_points] = interpolated

    return xa.DataArray(
        filled_values, dims=data.dims, coords=data.coords, attrs=data.attrs
    )

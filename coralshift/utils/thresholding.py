# general
import numpy as np
import xarray as xa

# plotting
import matplotlib.pyplot as plt


def calc_ts_stats(da: xa.DataArray) -> dict:
    """
    Compute statistics across spatial dimensions (mean, percentiles, min, max)

    Args:
        da (xa.DataArray): DataArray with "latitude", "longitude", and "time" dimensions

    Returns:
        dict: Dictionary with mean, 10th, 90th, 25th, 75th, min, and max values
    """
    # rechunk to allow calculation
    da = da.chunk({"time": -1})
    # select first depth if depth present TODO: a bit janky
    if "depth" in da.dims:
        da = da.isel(depth=0)
    stats = {
        "mean": da.mean(dim=["latitude", "longitude", "time"]).values,
        "10th": da.quantile(0.1, dim=["latitude", "longitude", "time"]).values,
        "90th": da.quantile(0.9, dim=["latitude", "longitude", "time"]).values,
        "25th": da.quantile(0.25, dim=["latitude", "longitude", "time"]).values,
        "75th": da.quantile(0.75, dim=["latitude", "longitude", "time"]).values,
        "min": da.min(dim=["latitude", "longitude", "time"]).values,
        "max": da.max(dim=["latitude", "longitude", "time"]).values,
    }
    return stats


def plot_timeseries_with_quantiles(
    ds: xa.DataArray, ylabel: str, **kwargs
) -> plt.Figure:
    """
    Plot a time series with mean, min, max, and 10th, 25th, 75th, and 90th percentiles

    Args:
        ds (xa.DataArray): DataArray with time series values
        ylabel (str): Label for the y-axis
        **kwargs: Additional keyword arguments for customizing the plot

    Returns:
        plt.Figure: Matplotlib figure object
    """
    stats = calc_ts_stats(ds)

    # Set default plotting parameters
    default_params = {
        "figsize": (12, 6),
        "dpi": 100,
        "color_min_max": "grey",
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

    # Plot the time series
    fig = plt.figure(figsize=plot_params["figsize"], dpi=plot_params["dpi"])
    time = ds["time"].values

    # Plot min and max as faint dotted lines
    plt.plot(
        time,
        stats["min"],
        color=plot_params["color_min_max"],
        linestyle="-",
        alpha=plot_params["alpha_min_max"],
    )
    plt.plot(
        time,
        stats["max"],
        color=plot_params["color_min_max"],
        linestyle="-",
        alpha=plot_params["alpha_min_max"],
    )

    # Fill between 10th and 90th percentiles
    plt.fill_between(
        time,
        stats["10th"],
        stats["90th"],
        color=plot_params["color_fill_90th"],
        alpha=plot_params["alpha_fill_90th"],
        label="90th percentiles",
    )

    # Fill between 25th and 75th percentiles
    plt.fill_between(
        time,
        stats["25th"],
        stats["75th"],
        color=plot_params["color_fill_50th"],
        alpha=plot_params["alpha_fill_50th"],
        label="50th percentiles",
    )

    # Plot mean as a thick line
    plt.plot(
        time,
        stats["mean"],
        color=plot_params["color_mean"],
        linewidth=plot_params["linewidth_mean"],
        label="Mean",
    )

    # Customize plot appearance
    plt.xlabel("Time", fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()

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
    return (rsdo_da * np.exp(kd490_da * -bathymetry_da)).to_dataset(name="par")

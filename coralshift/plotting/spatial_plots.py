import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import xarray as xa
import cartopy.crs as ccrs
import numpy as np

from coralshift.processing import data


def format_spatial_plot(image: xa.DataArray, fig: Figure, ax: Axes, title: str) -> None:
    """Format a spatial plot with a colorbar, title, coastlines, and gridlines.

    Parameters
    ----------
        image (xa.DataArray): image data to plot.
        fig (Figure): figure object to plot onto.
        ax (Axes): axes object to plot onto.
        title (str): title of the plot.

    Returns
    -------
        None
    """
    # great info here: https://stackoverflow.com/questions/13310594/positioning-the-colorbar
    fig.colorbar(image, orientation="horizontal", pad=0.1, label="elevation")
    ax.set_title(title)
    ax.coastlines(resolution="10m", color="black", linewidth=1)
    ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)


def plot_DEM(
    region_array,
    title: str,
    vmin: float = -50,
    vmax: float = 50,
    cmap="BrBG",
    landmask: bool = True,
) -> tuple[Figure, Axes]:
    """TODO: docstring"""
    # TODO: add option to plot satellite imagery for land instead of DEM
    map_proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection=map_proj)

    if landmask:
        vmax = 0

    im = region_array.plot(
        ax=ax,
        cmap="BrBG",
        # vmin=float(region_array.min().values), vmax=float(region_array.max().values),
        vmin=vmin,
        vmax=vmax,
        add_colorbar=False,
    )

    format_spatial_plot(im, fig, ax, title)

    return fig, ax, im


def plot_array_hist(
    array: tuple[xa.DataArray, np.ndarray],
    xlabel: str,
    ylabel: str,
    title: str,
    n_bins: int = 100,
) -> tuple[Figure, Axes]:
    """Plot a histogram of values in the input array.

    Parameters
    ----------
        array (tuple[xa.DataArray, np.ndarray]): tuple containing the input data as either a DataArray or numpy array.
        xlabel (str): label for the x-axis of the plot.
        ylabel (str): label for the y-axis of the plot.
        title (str): title of the plot.
        n_bins (int): number of bins to use in the histogram (default 100).

    Returns:
        A tuple containing the Figure and Axes objects of the plot.
    """
    fig, ax = plt.subplots()

    if not type(array) == np.ndarray:
        # if not np array, should be DataArray: so try fetching values. Index first dimension since 1xmxn array
        array = array.values[0]

    counts, bins = np.histogram(array, n_bins)
    ax.bar(bins[:-1], counts, width=np.diff(bins), align="edge")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    return fig, ax


def plot_vars_at_time(xa_ds: xa.Dataset, time: str = "2020-12-16T12:00:00"):
    """Plots the values of all non-empty variables in the given xarray Dataset at a specified time.

    Parameters
    ----------
    xa_ds (xa.Dataset): The xarray Dataset to plot.
    time (str, defaults to 2020-12-16T12:00:00): The time at which variables are plotted.
    TODO: add in satellite/ground values
    """
    non_empty_vars = data.return_non_empty_vars(xa_ds)
    blank_list = list(set(list(xa_ds.data_vars)) - set(non_empty_vars))
    print(
        f"The following variables returned empty arrays, and so are not plotted: {blank_list}"
    )

    num_plots = len(non_empty_vars)
    fig, axes = plt.subplots(
        nrows=num_plots, sharex=False, sharey=True, figsize=(10, num_plots * 6)
    )

    for i, (var_name, var) in enumerate(xa_ds[non_empty_vars].items()):
        var.sel(time=time).plot(ax=axes[i])
        axes[i].set_title(var_name)
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")
        axes[i].set_aspect("equal")
        # axes[i].add_feature(cfeature.COASTLINE)
        # axes[i].coastlines('50m')
    plt.suptitle(f"Time: {time}")

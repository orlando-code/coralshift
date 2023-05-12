import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib import animation
from pathlib import Path
import xarray as xa
import cartopy.crs as ccrs

# import cartopy.feature as cfeature
# import cartopy.mpl.ticker as cticker

import numpy as np

from coralshift.processing import data
from coralshift.utils import file_ops


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


def duration_to_interval(num_frames: int, duration: int = 5000) -> int:
    """Given the number of frames and duration of a gif in milliseconds, calculates the frame interval.

    Parameters
    ----------
    num_frames (int): The number of frames in the gif.
    duration (int, optional): The duration of the gif in milliseconds. Default is 5000ms (5 seconds).

    Returns
    -------
    int: The frame interval for the gif in milliseconds.
    """
    return duration / num_frames


def generate_variable_timeseries_gif(
    xa_da: xa.DataArray,
    start_end_freq: tuple[str, str, str] = None,
    variable_dict: dict = None,
    interval: int = 500,
    duration: int = None,
    repeat_delay: int = 5000,
    dest_dir_path: Path | str = None,
) -> animation.FuncAnimation:
    """Create an animation showing the values of a variable in an xarray DataArray over time.

    Parameters
    ----------
    xa_da (xr.DataArray): The xarray data array containing the variable to animate.
    start_end_freq (tuple of str, optional) A tuple containing the start date, end date, and frequency to index the
        "time" coordinate. If not provided, the whole time coordinate will be used.
    variable_dict (dict, optional): A dictionary with keys as the original variable names and values as the names to be
        displayed in the plot. If not provided, the original variable name will be used.
    interval (int, optional): The delay between frames in milliseconds.
    duration (int, optional): The duration of the GIF in milliseconds. If provided, the frame interval will be
        calculated automatically based on the number of frames and the duration.
    repeat_delay (int, optional): The delay before the animation repeats in milliseconds.
    dest_dir_path (Union[pathlib.Path, str], optional): The directory to save the output GIF. If not provided, the
        current working directory will be used.

    Returns
    -------
    animation.FuncAnimation: The animation object.

    TODO: plot formatting (coastlines and lat/lons). The issue at the moment is using imshow to plot. May be a way to
    just use the xa.plot() function on the axes
    More pressingly, fix the time resampling function
    """
    arrays = xa_da.values
    variable_name = xa_da.name

    if start_end_freq:
        # temporally resample DataArray
        xa_da, freq = data.resample_dataarray(xa_da, start_end_freq)
    # generate gif_name
    (start, end) = (
        data.date_from_dt(xa_da.time.min().values),
        data.date_from_dt(xa_da.time.max().values),
    )
    gif_name = f"{variable_name}_{start}_{end}"

    # fetch latitude and longitude bounds
    # coord_lims_dict = data.dict_xarray_coord_limits(xa_da)
    # lat_lims, lon_lims = coord_lims_dict["latitude"], coord_lims_dict["longitude"]

    fig, ax = plt.subplots(
        # subplot_kw={"projection": ccrs.PlateCarree()}
    )

    def update(i, variable_name=variable_name):
        ax.imshow(arrays[i], origin="lower", cmap="viridis")

        timestamp = data.date_from_dt(xa_da.time[i].values)
        if variable_dict:
            variable_name = variable_dict[variable_name]
        ax.set_title(f"{variable_name}\n{timestamp}")
        ax.axis("off")
        ax.set_aspect("equal")

        # # Longitude labels
        # ax.set_xticks(
        #     np.arange(lon_lims[0], lon_lims[1], 5), crs=ccrs.PlateCarree()
        # )
        # lon_formatter = cticker.LongitudeFormatter()
        # ax.xaxis.set_major_formatter(lon_formatter)

        # # Latitude labels
        # ax.set_yticks(
        #     np.arange(lat_lims[0], lat_lims[1], 5), crs=ccrs.PlateCarree()
        # )
        # lat_formatter = cticker.LatitudeFormatter()
        # ax.yaxis.set_major_formatter(lat_formatter)

        # ax.add_feature(cfeature.COASTLINE, color="r")

    fig.tight_layout()
    # if duration_specified
    if duration:
        interval = duration_to_interval(num_frames=xa_da.time.size, duration=duration)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=xa_da.time.size,
        interval=interval,
        repeat_delay=repeat_delay,
        repeat=True,
    )

    # save animation as a gif
    writergif = animation.PillowWriter(fps=30)
    # if destination directory not provided, save in current working directory
    if not dest_dir_path:
        dest_dir_path = file_ops.guarantee_existence(
            Path().absolute() / "gif_directory"
        )
    save_path = (dest_dir_path / gif_name).with_suffix(".gif")

    ani.save(str(save_path), writer=writergif)
    print(
        f"""Gif for {variable_name} between {start} and {end} written to {save_path}.
        \nInterval: {interval}, repeat_delay: {repeat_delay}."""
    )

    return ani

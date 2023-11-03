from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib import animation, colors
import matplotlib.gridspec as gridspec

from pathlib import Path
import xarray as xa
import cartopy.crs as ccrs

import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker

import numpy as np
from tqdm import tqdm

from coralshift.processing import spatial_data
from coralshift.dataloading import bathymetry
from coralshift.utils import file_ops


def hex_to_rgb(value):
    """
    Convert a hexadecimal color code to RGB values.

    Parameters
    ----------
    value (str): The hexadecimal color code as a string of 6 characters.

    Returns
    -------
    tuple: A tuple of three RGB values.
    """
    value = value.strip("#")  # removes hash symbol if present
    hex_el = len(value)
    return tuple(
        int(value[i : i + hex_el // 3], 16)  # noqa
        for i in range(0, hex_el, hex_el // 3)
    )


def rgb_to_dec(value):
    """
    Convert RGB color values to decimal values (each value divided by 256).

    Parameters
    ----------
    value (list): A list of three RGB values.

    Returns
    -------
    list: A list of three decimal values.
    """
    return [v / 256 for v in value]


def get_continuous_cmap(hex_list, float_list=None):
    """
    Create and return a color map that can be used in heat map figures.

    Parameters
    ----------
    hex_list (list of str): List of hex code strings representing colors.
    float_list (list of float, optional): List of floats between 0 and 1, same length as hex_list. Must start with 0
        and end with 1.

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap: The created color map.
    """
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0, 1, len(rgb_list)))

    cdict = dict()
    for num, col in enumerate(["red", "green", "blue"]):
        col_list = [
            [float_list[i], rgb_list[i][num], rgb_list[i][num]]
            for i in range(len(float_list))
        ]
        cdict[col] = col_list
    cmp = colors.LinearSegmentedColormap("my_cmp", segmentdata=cdict, N=256)
    return cmp


def get_cbar(cbar_type: str = "seq"):
    """
    Get a colormap for colorbar based on the specified type.

    Parameters
    ----------
    cbar_type (str, optional): The type of colormap to retrieve. Options are 'seq' for sequential colormap and 'div'
        for diverging colormap.

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap: The colormap object.
    """
    if cbar_type == "seq":
        return get_continuous_cmap(
            ["#3B9AB2", "#78B7C5", "#EBCC2A", "#E1AF00", "#d83c04"]
        )
    elif cbar_type == "div":
        return get_continuous_cmap(
            ["#3B9AB2", "#78B7C5", "#FFFFFF", "#E1AF00", "#d83c04"]
        )
    elif cbar_type == "lim":
        return get_continuous_cmap(["#78B7C5", "#EBCC2A", "#E1AF00"])
    else:
        raise ValueError(f"{cbar_type} not recognised.")


def plot_spatial(
    xa_da: xa.DataArray,
    fax: Axes = None,
    title: str = None,
    name: str = None,
    figsize: tuple[float, float] = (10, 10),
    val_lims: tuple[float, float] = None,
    cmap_type: str = "seq",
    symmetric: bool = False,
    edgecolor: str = "black",
    cbar: bool = True,
    orient_colorbar: str = "vertical",
    cbar_pad: float = 0.1,
) -> tuple[Figure, Axes]:
    """
    Plot a spatial plot with colorbar, coastlines, landmasses, and gridlines.

    Parameters
    ----------
    xa_da (xa.DataArray): The input xarray DataArray representing the spatial data.
    title (str, optional): The title of the plot.
    name (str, optional): The name of the DataArray.
    val_lims (tuple[float, float], optional): The limits of the colorbar range.
    cmap_type (str, optional): The type of colormap to use.
    symmetric (bool, optional): Whether to make the colorbar symmetric around zero.
    edgecolor (str, optional): The edge color of the landmasses.
    orient_colorbar (str, optional): The orientation of the colorbar ('vertical' or 'horizontal').

    Returns
    -------
    tuple: The figure and axes objects.
    """
    map_proj = ccrs.PlateCarree()
    # may need to change this

    if not fax:
        fig = plt.figure(figsize=figsize)
        ax = plt.axes(projection=map_proj)
    else:
        fig = fax[0]
        ax = fax[1]

    resolution_d = np.mean(spatial_data.calculate_spatial_resolution(xa_da))
    resolution_m = np.mean(spatial_data.degrees_to_distances(resolution_d))

    # if not name:
    if isinstance(xa_da, xa.DataArray):
        name = xa_da.name
    else:
        name = list(xa_da.data_vars)[0]
    if not title:
        title = name + " at {:.4f}° (~{:.0f} m) resolution".format(  # noqa
            resolution_d, resolution_m
        )

    # if colorbar limits not specified, set to be maximum of array
    if not val_lims:
        vmin, vmax = np.nanmin(xa_da.values), np.nanmax(xa_da.values)
    else:
        vmin, vmax = min(val_lims), max(val_lims)

    cmap = get_cbar(cmap_type)

    # if choosing coloorbar to be centred arouoond zero (e.g. for highlighting residuals)
    if symmetric:
        vmin, vmax = (-vmax, vmax) if abs(vmin) > abs(vmax) else (vmin, -vmin)

    im = xa_da.plot(ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, add_colorbar=False)
    # nicely format spatial plot
    format_spatial_plot(
        image=im,
        fig=fig,
        ax=ax,
        title=title,
        name=name,
        cbar=cbar,
        orient_colorbar=orient_colorbar,
        cbar_pad=cbar_pad,
        edgecolor=edgecolor,
    )

    return fig, ax, im


def format_spatial_plot(
    image: xa.DataArray,
    fig: Figure,
    ax: Axes,
    title: str = "",
    name: str = "",
    cbar: bool = True,
    orient_colorbar: str = "horizontal",
    cbar_pad: float = 0.1,
    edgecolor: str = "black",
) -> tuple[Figure, Axes]:
    """Format a spatial plot with a colorbar, title, coastlines and landmasses, and gridlines.

    Parameters
    ----------
        image (xa.DataArray): image data to plot.
        fig (Figure): figure object to plot onto.
        ax (Axes): axes object to plot onto.
        title (str): title of the plot.

    Returns
    -------
        Figure, Axes
    """
    if cbar:
        plt.colorbar(
            image, orientation=orient_colorbar, label=name, pad=cbar_pad, fraction=0.046
        )
    ax.set_title(title)
    # ax.coastlines(resolution="10m", color="red", linewidth=1)
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            "physical", "land", "10m", edgecolor=edgecolor, facecolor="#d2ccc4"
        )
    )
    ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
    # ax.gridlines(draw_labels={"bottom": "x", "left": "y"})

    return fig, ax


def plot_spatial_diffs(
    xa_d_pred: xa.DataArray,
    xa_d_gt: xa.DataArray,
    figsize: tuple[float, float] = (16, 9),
) -> None:
    """
    Plot the spatial differences between predicted and ground truth data.

    Parameters
    ----------
        xa_d_pred (xa.DataArray): Predicted data.
        xa_d_gt (xa.DataArray): Ground truth data.
        figsize (tuple[float, float], optional): Figure size. Default is (16, 9).

    Returns
    -------
        None
    """
    xa_diff = (xa_d_gt - xa_d_pred).rename("predicted/gt_residuals")

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 2)

    # left plot
    ax_r = fig.add_subplot(gs[:, 0], projection=ccrs.PlateCarree())
    plot_spatial(fax=(fig, ax_r), xa_da=xa_diff, cmap_type="div", symmetric=True)

    # right plots
    ax_l_t = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
    plot_spatial(xa_d_gt, fax=(fig, ax_l_t), title="ground truth")
    ax_l_b = fig.add_subplot(gs[1, 1], projection=ccrs.PlateCarree())
    plot_spatial(xa_d_pred, fax=(fig, ax_l_b), title="inferred label")


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

    if not isinstance(array, np.ndarray):
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


def generate_like_variable_timeseries_gifs(
    xa_ds: xa.Dataset,
    variables: list[str] = None,
    start_end_freq: tuple[str, str, str] = None,
    variable_dict: dict = None,
    interval: int = 500,
    duration: int = None,
    repeat_delay: int = 5000,
    dest_dir_path: Path | str = None,
) -> dict:
    """Wrapper for generate_variable_timeseries_gif allowing generation of a set of similar gifs for each specified
    variable in an xarray Dataset

    Parameters
    ----------
    xa_das (xr.Dataset): The xarray Dataset array containing variables to animate.
    variables (list[str], optional): Choice of variables to animate. Defaults to None (all variablees animated).
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
    list[animation.FuncAnimation]: list containing the animation objects.
    """
    # if no specific variables specified, animate all available
    if not variables:
        variables = list(xa_ds.data_vars)

    ani_dict = {}
    for var in tqdm(variables):
        ani = generate_variable_timeseries_gif(
            xa_ds[var],
            start_end_freq=start_end_freq,
            variable_dict=variable_dict,
            interval=interval,
            duration=duration,
            repeat_delay=repeat_delay,
            dest_dir_path=dest_dir_path,
        )
        ani_dict[var] = ani

    return ani_dict


def format_xa_array_spatial_plot(
    ax: Axes, xa_da: xa.DataArray, variable_dict: dict = None, coastlines: bool = True
) -> tuple[Axes, str, tuple[float], tuple[float]]:
    """Formats a Matplotlib axes object for a spatial plot of an xarray DataArray.

    Parameters:
    ax (Axes): The Matplotlib axes object to format.
    xa_da (xr.DataArray): The xarray DataArray to plot.
    coastlines (bool): Whether to include coastlines on the plot. Defaults to True.

    Returns:
    Tuple[Axes, Tuple[float, float], Tuple[float, float]]: A tuple containing:
        - The formatted Matplotlib axes object.
        - A tuple of the minimum and maximum latitude values for the plot.
    """
    # set title
    variable_name = xa_da.name
    if variable_dict:
        variable_name = variable_dict[variable_name]
    ax.set_title(variable_name)

    # determine minimum and maximum coordinates
    coord_lims_dict = spatial_data.dict_xarray_coord_limits(xa_da)
    lat_lims, lon_lims = coord_lims_dict["latitude"], coord_lims_dict["longitude"]

    # set longitude labels
    ax.set_xticks(np.arange(lon_lims[0], lon_lims[1], 5), crs=ccrs.PlateCarree())
    lon_formatter = cticker.LongitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)

    # set latitude labels
    ax.set_yticks(np.arange(lat_lims[0], lat_lims[1], 5), crs=ccrs.PlateCarree())
    lat_formatter = cticker.LatitudeFormatter()
    ax.yaxis.set_major_formatter(lat_formatter)

    if coastlines:
        ax.add_feature(cfeature.COASTLINE, edgecolor="r", linewidth=0.5)
    ax.set_extent([lon_lims[0], lon_lims[1], lat_lims[0], lat_lims[1]])
    ax.set_aspect("equal")

    return ax, variable_name, lat_lims, lon_lims


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
    xa_da (xr.DataArray): The xarray DataArray containing the variable to animate.
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

    TODO: Fix the time resampling function
    """
    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
    ax, variable_name, _, _ = format_xa_array_spatial_plot(ax, xa_da)
    fig.tight_layout()

    def update(i, variable_name=variable_name) -> None:
        """Updates a Matplotlib plot with a new frame.

        Parameters
        ----------
        i (int): The index of the frame to display.
        variable_name (str): The name of the variable being plotted.

        Returns
        -------
        None
        TODO: add single colorbar (could delete previous one each time, somehow)
        """
        cb = True
        timestamp = spatial_data.date_from_dt(xa_da.time[i].values)
        ax.set_title(f"{variable_name}\n{timestamp}")
        if i > 1:
            cb = False
        xa_da.isel(time=i).plot(ax=ax, add_colorbar=cb, cmap="viridis")

    if start_end_freq:
        # temporally resample DataArray
        xa_da, freq = spatial_data.resample_dataarray(xa_da, start_end_freq)

    # generate gif_name
    (start, end) = (
        spatial_data.date_from_dt(xa_da.time.min().values),
        spatial_data.date_from_dt(xa_da.time.max().values),
    )
    gif_name = f"{variable_name}_{start}_{end}"

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
        blit=False,
    )
    # TODO: put in different function
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


def plot_var(
    xa_da: xa.DataArray,
    figsize: tuple[float, float] = (10, 6),
    variable_dict: dict = None,
    coastlines: bool = True,
):
    fig, ax = plt.subplots(
        figsize=figsize,
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    xa_da.plot(ax=ax)

    format_xa_array_spatial_plot(ax, xa_da, coastlines=coastlines)
    return fig, ax


def plot_var_at_time(
    xa_da: xa.DataArray,
    time: str = None,
    figsize: tuple[float, float] = (10, 6),
    variable_dict: dict = None,
    coastlines: bool = True,
):
    """Plots the values of a variable in a single xarray DataArray at a specified time.

    Parameters
    ----------
    var (xa.DataArray): The xarray DataArray to plot.
    time (str): The time at which variables are plotted.
    variable_name (str): The name of the variable being plotted.
    lat_lims (tuple): tuple of latitude bounds.
    lon_lims (tuple): tuple of longitude bounds.
    """
    fig, ax = plt.subplots(
        figsize=figsize,
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    format_xa_array_spatial_plot(ax, xa_da, coastlines=coastlines)

    # if time not specified, choose latest
    if not time:
        time = xa_da.time.max().values

    xa_da.sel(time=time).plot(ax=ax)

    return fig, ax


def plot_vars_at_time(
    xa_ds: xa.Dataset,
    time: str = None,
    figsize: tuple[float, float] = (10, 6),
    variable_dict: dict = None,
    coastlines: bool = True,
):
    """Plots the values of all non-empty variables in the given xarray Dataset at a specified time.

    Parameters
    ----------
    xa_ds (xa.Dataset): The xarray Dataset to plot.
    time (str, defaults to 2020-12-16T12:00:00): The time at which variables are plotted.
    TODO: add in satellite/ground values
    """
    non_empty_vars = spatial_data.return_non_empty_vars(xa_ds)
    blank_list = list(set(list(xa_ds.data_vars)) - set(non_empty_vars))
    if len(blank_list) > 0:
        print(
            f"The following variables returned empty arrays, and so are not plotted: {blank_list}"
        )

    for var_name, xa_da in tqdm(xa_ds[non_empty_vars].items()):
        plot_var_at_time(
            xa_da,
            time=time,
            figsize=figsize,
            variable_dict=variable_dict,
            coastlines=coastlines,
        )


def visualise_variable_in_region(
    xa_array: xa.DataArray,
    lat_lims: tuple[float, float] = None,
    lon_lims: tuple[(float, float)] = None,
) -> None:
    """
    Visualizes a variable in a specified geographical region.

    Parameters:
        xa_array (xarray.DataArray): Input data array containing the variable to visualize.
        lat_lims (tuple): Latitude limits as (min_lat, max_lat).
        lon_lims (tuple): Longitude limits as (min_lon, max_lon).

    Returns:
        None
    """
    # TODO: fix flat_data
    fig, ax = plt.subplots(
        1,
        3,
        figsize=[15, 5],
        # sharey=True,
        gridspec_kw={"width_ratios": [1, 2, 1]},
    )

    if (lat_lims and lon_lims) is not None:
        xa_array = xa_array.sel(
            latitude=slice(min(lat_lims), max(lat_lims)),
            longitude=slice(min(lon_lims), max(lon_lims)),
        )

    # flat_data = xa_array.values.flatten()
    spatial_mean = xa_array.mean(dim=["latitude", "longitude"])
    ax[0].boxplot(spatial_mean)
    spatial_mean.plot.line(ax=ax[1])
    spatial_mean.plot.hist(ax=ax[2])

    # format
    ax[0].set_ylabel(xa_array.name)
    ax[1].set_xlabel("time")

    return spatial_mean


def plot_reef_areas(region: list[str] = None):
    reef_areas = bathymetry.ReefAreas()
    lats, lons, rectangles, names = [], [], [], []

    if region:
        for region_name in region:
            name = reef_areas.get_letter(region_name)
            dataset = reef_areas.get_dataset(name)
            if dataset:
                lat_pair, lon_pair = dataset["lat_range"], dataset["lon_range"]
                lats.append(lat_pair)
                lons.append(lon_pair)

                # Add a rectangle at the specified coordinates (in front of overlay)
                rect = plt.Rectangle(
                    (min(lon_pair), min(lat_pair)),
                    abs(max(lon_pair) - min(lon_pair)),
                    abs(max(lat_pair) - min(lat_pair)),
                    edgecolor="#f90202",
                    facecolor="none",
                    zorder=10,
                )
                rectangles.append(rect)
                names.append(name)
    else:
        for region_ds in reef_areas.datasets:
            name = reef_areas.get_letter(region_ds["short_name"])
            (lat_pair, lon_pair) = reef_areas.get_lat_lon_limits(name)
            lats.append(lat_pair)
            lons.append(lon_pair)

            # Add a rectangle at the specified coordinates (in front of overlay)
            rect = plt.Rectangle(
                (min(lon_pair), min(lat_pair)),
                abs(max(lon_pair) - min(lon_pair)),
                abs(max(lat_pair) - min(lat_pair)),
                edgecolor="#f90202",
                facecolor="none",
                zorder=10,
            )
            rectangles.append(rect)
            names.append(name)

    # Create a GeoAxes with desired projection
    projection = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection=projection))

    # Set the extent of the map depending on areas included (with some padding)
    ax.set_extent(
        [
            min(lons, key=lambda x: x[0])[0] - 0.5,
            max(lons, key=lambda x: x[0])[1] + 0.5,
            min(lats, key=lambda x: x[0])[1] - 2,
            max(lats, key=lambda x: x[0])[0] + 2,
        ],
        crs=projection,
    )

    # plot and format underlay
    # add land feature (colours map those used in report)
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            "physical", "land", "10m", edgecolor="black", facecolor="#d2ccc4"
        )
    )
    ax.gridlines(draw_labels=True, linestyle="-", linewidth=0.5, color="gray")

    for rect, name in zip(rectangles, names):
        ax.add_patch(rect)
        rect_center = rect.get_bbox().get_points().mean(axis=0)

        # Add a small white square behind the letter
        square_size = 1.0  # 1 degree side length
        square = plt.Rectangle(
            (rect_center[0] - square_size / 2, rect_center[1] - square_size / 2),
            square_size,
            square_size,
            edgecolor="none",
            facecolor="white",
            alpha=1,
            zorder=9,
        )
        ax.add_patch(square)

        ax.text(
            rect_center[0],
            rect_center[1],
            name,
            ha="center",
            va="center",
            fontsize=1.5 * plt.rcParams["font.size"],
            fontweight="bold",
            zorder=10,
        )

    plt.title("Selected region(s) of interest")


def plot_var_mask(
    xa_d: xa.Dataset | xa.DataArray,
    limits: tuple[float] = [-2000, 0],
    title: str = "masked variable",
) -> None:
    # plot shallow water mask
    shallow_mask = spatial_data.generate_var_mask(xa_d)
    plot_spatial(shallow_mask, cmap_type="lim", title=title, cbar=False)
    return shallow_mask

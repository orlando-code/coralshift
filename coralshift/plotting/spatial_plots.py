# plotting
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib import animation, colors
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import seaborn as sns

# general
import pandas as pd
import numpy as np

# file ops
from pathlib import Path
from tqdm import tqdm

# spatial
import xarray as xa
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
from rasterio import enums

# custom
from coralshift.processing import spatial_data
from coralshift.dataloading import bathymetry
from coralshift.utils import file_ops


# TODO: not working universally
def customize_plot_colors(fig, ax, background_color="#212121", text_color="white"):
    # Set figure background color
    fig.patch.set_facecolor(background_color)

    # Set axis background color (if needed)
    ax.set_facecolor(background_color)

    # Set text color for all elements in the plot
    for text in fig.texts:
        text.set_color(text_color)
    for text in ax.texts:
        text.set_color(text_color)
    for text in ax.xaxis.get_ticklabels():
        text.set_color(text_color)
    for text in ax.yaxis.get_ticklabels():
        text.set_color(text_color)
    ax.title.set_color(text_color)
    ax.xaxis.label.set_color(text_color)
    ax.yaxis.label.set_color(text_color)

    # Set legend text color
    legend = ax.get_legend()
    if legend:
        for text in legend.get_texts():
            text.set_color(text_color)
    # # set cbar labels
    # cbar = ax.collections[0].colorbar
    # cbar.set_label(color=text_color)
    # cbar.ax.yaxis.label.set_color(text_color)
    ax.tick_params(axis="x", colors="white")
    ax.tick_params(axis="y", colors="white")

    return fig, ax


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


# TODO: wrap up colourmaps into a class
# class ColourMaps():
#     __init__(self):
#         self.sequential_hexes = ["#3B9AB2", "#78B7C5", "#EBCC2A", "#E1AF00", "#d83c04"],
#         self.diverging_hexes = ["#3B9AB2", "#78B7C5", "#FFFFFF", "#E1AF00", "#d83c04"]

#     def get_cmap(self, cmap_type: str = "seq"):


def get_n_colors_from_hexes(
    num: int,
    hex_list: list[str] = ["#3B9AB2", "#78B7C5", "#EBCC2A", "#E1AF00", "#d83c04"],
) -> list[str]:
    """
    from Wes Anderson: https://github.com/karthik/wesanderson/blob/master/R/colors.R
    Get a list of n colors from a list of hex codes.

    Args:
        num (int): The number of colors to return.
        hex_list (list[str]): The list of hex codes from which to create spectrum for sampling.

    Returns:
        list[str]: A list of n hex codes.
    """
    cmap = get_continuous_cmap(hex_list)
    colors = [cmap(i / num) for i in range(num)]
    hex_codes = [mcolors.to_hex(color) for color in colors]
    return hex_codes


class ColourMapGenerator:
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

    def __init__(self):
        self.sequential_hexes = ["#3B9AB2", "#78B7C5", "#EBCC2A", "#E1AF00", "#d83c04"]
        self.diverging_hexes = ["#3B9AB2", "#78B7C5", "#FFFFFF", "#E1AF00", "#d83c04"]
        self.cyclical_hexes = [
            "#3B9AB2",
            "#78B7C5",
            "#EBCC2A",
            "#E1AF00",
            "#d83c04",
            "#E1AF00",
            "#EBCC2A",
            "#78B7C5",
            "#3B9AB2",
        ]
        self.conf_mat_hexes = ["#EEEEEE", "#3B9AB2", "#cae7ed", "#d83c04", "#E1AF00"]
        self.residual_hexes = ["#3B9AB2", "#78B7C5", "#fafbfc", "#E1AF00", "#d83c04"]
        self.lim_red_hexes = ["#EBCC2A", "#E1AF00", "#d83c04"]
        self.lim_blue_hexes = ["#3B9AB2", "#78B7C5", "#FFFFFF"]

    def get_cmap(self, cbar_type, vmin=None, vmax=None):
        if cbar_type == "seq":
            return get_continuous_cmap(self.sequential_hexes)
        if cbar_type == "inc":
            return get_continuous_cmap(self.sequential_hexes[2:])
        elif cbar_type == "div":
            if not (vmin and vmax):
                raise ValueError(
                    "Minimum and maximum values needed for divergent colorbar"
                )
            cmap = get_continuous_cmap(self.diverging_hexes)
            norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
            return cmap, norm
            # return get_continuous_cmap(self.diverging_hexes)
        elif cbar_type == "res":
            if not (vmin and vmax):
                raise ValueError(
                    "Minimum and maximum values needed for divergent colorbar"
                )
            cmap = get_continuous_cmap(self.residual_hexes)
            norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
            return cmap, norm
        elif cbar_type == "cyc":
            return get_continuous_cmap(self.cyclical_hexes)
        elif cbar_type == "lim_blue":
            return get_continuous_cmap(self.lim_blue_hexes)
        elif cbar_type == "lim_red":
            return get_continuous_cmap(self.lim_red_hexes)
        elif cbar_type == "spatial_conf_matrix":
            return mcolors.ListedColormap(self.conf_mat_hexes)
        else:
            raise ValueError(f"{cbar_type} not recognised.")


def generate_geo_axis(
    figsize: tuple[float, float] = (10, 10), map_proj=ccrs.PlateCarree(), dpi=300
):
    return plt.figure(figsize=figsize, dpi=dpi), plt.axes(projection=map_proj)


def plot_spatial(
    xa_da: xa.DataArray,
    fax: Axes = None,
    title: str = "default",
    figsize: tuple[float, float] = (10, 10),
    val_lims: tuple[float, float] = None,
    presentation_format: bool = False,
    labels: list[str] = ["l", "b"],
    cbar_dict: dict = None,
    cartopy_dict: dict = None,
    label_style_dict: dict = None,
    map_proj=ccrs.PlateCarree(),
    alpha: float = 1,
    extent: list[float] = None,
) -> tuple[Figure, Axes]:
    """
    Plot a spatial plot with colorbar, coastlines, landmasses, and gridlines.

    Parameters
    ----------
    xa_da (xa.DataArray): The input xarray DataArray representing the spatial data.
    title (str, optional): The title of the plot.
    cbar_name (str, optional): The name of the DataArray.
    val_lims (tuple[float, float], optional): The limits of the colorbar range.
    cmap_type (str, optional): The type of colormap to use.
    symmetric (bool, optional): Whether to make the colorbar symmetric around zero.
    edgecolor (str, optional): The edge color of the landmasses.
    orientation (str, optional): The orientation of the colorbar ('vertical' or 'horizontal').
    labels (list[str], optional): Which gridlines to include, as strings e.g. ["t","r","b","l"]
    map_proj (str, optional): The projection of the map.
    extent (list[float], optional): The extent of the plot as [min_lon, max_lon, min_lat, max_lat].

    Returns
    -------
    tuple: The figure and axes objects.
    TODO: saving option and tidy up presentation formatting
    """
    # may need to change this
    # for some reason fig not including axis ticks. Universal for other plotting
    if not fax:
        if extent == "global":
            fig, ax = generate_geo_axis(figsize=figsize, map_proj=ccrs.Robinson())
            ax.set_global()
        else:
            fig, ax = generate_geo_axis(figsize=figsize, map_proj=map_proj)

    else:
        fig, ax = fax[0], fax[1]

    if isinstance(extent, list):
        ax.set_extent(extent, crs=map_proj)

    default_cbar_dict = {
        "cbar_name": None,
        "cbar": True,
        "orientation": "vertical",
        "cbar_pad": 0.1,
        "cbar_frac": 0.025,
        "cmap_type": "seq",
    }

    if cbar_dict:
        for k, v in cbar_dict.items():
            default_cbar_dict[k] = v
        if val_lims:
            default_cbar_dict["extend"] = "both"

    # if not cbarn_name specified, make name of variable
    cbar_name = default_cbar_dict["cbar_name"]
    if isinstance(xa_da, xa.DataArray) and not cbar_name:
        cbar_name = xa_da.name

    # if title not specified, make title of variable at resolution
    if title:
        if title == "default":
            resolution_d = np.mean(spatial_data.calculate_spatial_resolution(xa_da))
            resolution_m = np.mean(spatial_data.degrees_to_distances(resolution_d))
            title = (
                f"{cbar_name} at {resolution_d:.4f}° (~{resolution_m:.0f} m) resolution"
            )

    # if colorbar limits not specified, set to be maximum of array
    if not val_lims:  # TODO: allow dynamic specification of only one of min/max
        vmin, vmax = np.nanmin(xa_da.values), np.nanmax(xa_da.values)
    else:
        vmin, vmax = min(val_lims), max(val_lims)

    if (
        default_cbar_dict["cmap_type"] == "div"
        or default_cbar_dict["cmap_type"] == "res"
    ):
        if vmax < 0:
            vmax = 0.01
        cmap, norm = ColourMapGenerator().get_cmap(
            default_cbar_dict["cmap_type"], vmin, vmax
        )
    else:
        cmap = ColourMapGenerator().get_cmap(default_cbar_dict["cmap_type"])

    im = xa_da.plot(
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        add_colorbar=False,  # for further formatting later
        transform=ccrs.PlateCarree(),
        alpha=alpha,
        norm=(
            norm
            if (
                default_cbar_dict["cmap_type"] == "div"
                or default_cbar_dict["cmap_type"] == "res"
            )
            else None
        ),
    )

    if presentation_format:
        fig, ax = customize_plot_colors(fig, ax)
        # ax.tick_params(axis="both", which="both", length=0)

    # nicely format spatial plot
    format_spatial_plot(
        image=im,
        fig=fig,
        ax=ax,
        title=title,
        # cbar_name=cbar_name,
        # cbar=default_cbar_dict["cbar"],
        # orientation=default_cbar_dict["orientation"],
        # cbar_pad=default_cbar_dict["cbar_pad"],
        # cbar_frac=default_cbar_dict["cbar_frac"],
        cartopy_dict=cartopy_dict,
        presentation_format=presentation_format,
        labels=labels,
        cbar_dict=default_cbar_dict,
        label_style_dict=label_style_dict,
    )

    return fig, ax, im


def format_cbar(image, fig, ax, cbar_dict, labels: list[str] = ["l", "b"]):

    if cbar_dict["orientation"] == "vertical":
        cbar_rect = [
            ax.get_position().x1 + 0.01,
            ax.get_position().y0,
            0.02,
            ax.get_position().height,
        ]
        labels = [el if el != "b" else "t" for el in labels or []]
    else:
        cbar_rect = [
            ax.get_position().x0,
            ax.get_position().y0 - 0.04,
            ax.get_position().width,
            0.02,
        ]
        labels = [el if el != "b" else "t" for el in labels or []]
    cax = fig.add_axes(cbar_rect)

    cb = plt.colorbar(
        image,
        orientation=cbar_dict["orientation"],
        label=cbar_dict["cbar_name"],
        cax=cax,
        extend=cbar_dict["extend"] if "extend" in cbar_dict else "neither",
    )
    if cbar_dict["orientation"] == "horizontal":
        cbar_ticks = cb.ax.get_xticklabels()
    else:
        cbar_ticks = cb.ax.get_yticklabels()

    return cb, cbar_ticks, labels


def format_cartopy_display(ax, cartopy_dict: dict = None):

    default_cartopy_dict = {
        "category": "physical",
        "name": "land",
        "scale": "10m",
        "edgecolor": "black",
        "facecolor": (0, 0, 0, 0),  # "none"
        "linewidth": 0.5,
        "alpha": 0.3,
    }

    if cartopy_dict:
        for k, v in cartopy_dict.items():
            default_cartopy_dict[k] = v

    ax.add_feature(
        cfeature.NaturalEarthFeature(
            default_cartopy_dict["category"],
            default_cartopy_dict["name"],
            default_cartopy_dict["scale"],
            edgecolor=default_cartopy_dict["edgecolor"],
            facecolor=default_cartopy_dict["facecolor"],
            linewidth=default_cartopy_dict["linewidth"],
            alpha=default_cartopy_dict["alpha"],
        )
    )

    return ax


def format_spatial_plot(
    image: xa.DataArray,
    fig: Figure,
    ax: Axes,
    title: str = None,
    # cbar: bool = True,
    # cmap_type: str = "seq",
    presentation_format: bool = False,
    labels: list[str] = ["l", "b"],
    cbar_dict: dict = None,
    cartopy_dict: dict = None,
    label_style_dict: dict = None,
) -> tuple[Figure, Axes]:
    """Format a spatial plot with a colorbar, title, coastlines and landmasses, and gridlines.

    Parameters
    ----------
        image (xa.DataArray): image data to plot.
        fig (Figure): figure object to plot onto.
        ax (Axes): axes object to plot onto.
        title (str): title of the plot.
        cbar_name (str): label of colorbar.
        cbar (bool): whether to include a colorbar.
        orientation (str): orientation of colorbar.
        cbar_pad (float): padding of colorbar.
        edgecolor (str): color of landmass edges.
        presentation_format (bool): whether to format for presentation.
        labels (list[str]): which gridlines to include, as strings e.g. ["t","r","b","l"]
        label_style_dict (dict): dictionary of label styles.

    Returns
    -------
        Figure, Axes
    """
    if cbar_dict and cbar_dict["cbar"]:
        cb, cbar_ticks, labels = format_cbar(image, fig, ax, cbar_dict, labels)

    ax = format_cartopy_display(ax, cartopy_dict)
    ax.set_title(title)

    # format ticks, gridlines, and colours
    ax.tick_params(axis="both", which="major")
    default_label_style_dict = {"fontsize": 12, "color": "black", "rotation": 45}

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        # x_inline=False, y_inline=False
    )

    if label_style_dict:
        for k, v in label_style_dict.items():
            default_label_style_dict[k] = v
    if presentation_format:
        default_label_style_dict["color"] = "white"
        if cbar_dict and cbar_dict["cbar"]:
            plt.setp(cbar_ticks, color="white")
            cb.set_label(cbar_dict["cbar_name"], color="white")

    gl.xlabel_style = default_label_style_dict
    gl.ylabel_style = default_label_style_dict

    if (
        not labels
    ):  # if no labels specified, set up something to iterate through returning nothing
        labels = [" "]
    if labels:
        # convert labels to relevant boolean: ["t","r","b","l"]
        gl.top_labels = "t" in labels
        gl.bottom_labels = "b" in labels
        gl.left_labels = "l" in labels
        gl.right_labels = "r" in labels

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
    plot_spatial(shallow_mask, cmap_type="lim_blue", title=title, cbar=False)
    return shallow_mask


def visualise_predictions(
    pred_df: pd.DataFrame, gt: str = "unep_coral_presence", title: str = None
) -> None:
    """
    Visualise predictions against ground truth.

    Args:
        pred_df (pd.DataFrame): The dataframe containing the predictions and ground truth.
        gt (str): The name of the ground truth variable.
        title (str): The title of the plot.

    Returns:
        None
    """
    f, ax = plt.subplots(
        ncols=3, figsize=(14, 5), subplot_kw={"projection": ccrs.PlateCarree()}
    )

    plot_spatial(
        pred_df["unep_coral_presence"].isel(time=0),
        title="Ground Truth",
        fax=[f, ax[0]],
    )
    plot_spatial(
        pred_df["prediction"].isel(time=0), title="Predictions", fax=[f, ax[1]]
    )
    ax[2].scatter(pred_df["unep_coral_presence"], pred_df["prediction"])

    # formatting
    ax[2].set_xlabel("Actual")
    ax[2].set_ylabel("Predicted")
    ax[2].set_title("Predictions vs Actual")
    ax[2].axline((0, 0), slope=1, c="k")
    ax[2].set_ylim(0, 1)
    if title:
        f.suptitle(title)


def plot_heatmap_from_dict_pairs(
    dictionary: dict,
    statistic: str,
    key1: str = "key 1",
    key2: str = "key 2",
    cbar_label: str = None,
):
    """
    Plot a heatmap from a dictionary of pairs.

    Args:
        dictionary (dict): A dictionary containing the data.
        statistic (str): The statistic to be plotted.
        key1 (str, optional): The label for the first key. Defaults to "key 1".
        key2 (str, optional): The label for the second key. Defaults to "key 2".
        cbar_label (str, optional): The label for the colorbar. If not provided, the statistic name will be used.

    Returns:
        None
    """
    av_iteration_times = {
        k1: np.mean([dictionary[k1][k2][statistic] for k2 in dictionary[k1].keys()])
        for k1 in dictionary.keys()
    }
    sorted_data = sorted(av_iteration_times, key=av_iteration_times.get)[::-1]

    heatmap_data = []
    for k1 in sorted_data:
        row = []
        for k2 in dictionary[k1].keys():
            row.append(dictionary[k1][k2][statistic])
        heatmap_data.append(row)

    # slightly hacky way to get rasterio resampling information
    if isinstance(sorted_data[0], enums.Resampling):
        sorted_data_names = [resampling.name for resampling in sorted_data]
    else:
        sorted_data_names = sorted_data

    heatmap_df = pd.DataFrame(
        heatmap_data, index=sorted_data_names, columns=dictionary[sorted_data[0]].keys()
    )

    plt.figure(figsize=(10, 8))

    if not cbar_label:
        cbar_label = statistic

    sns.heatmap(
        heatmap_df,
        annot=True,
        cmap=ColourMapGenerator().get_cmap("seq"),
        cbar_kws={"label": cbar_label},
    )
    plt.title(f"Heatmap of {statistic} for {key1} and {key2}")
    plt.xlabel(key2)
    plt.ylabel(key1)
    plt.show()


def plot_performance_against_key(timings_dict, k1_name, title: str = None):
    """
    Plot the average iteration duration against a key value.

    Parameters
    ----------
    timings_dict : dict
        A dictionary containing the timing data.
    k1_name : str
        The label for the key value.
    title : str, optional
        The title of the plot, by default None.

    Returns
    -------
    None
    """
    ks = [k for k in timings_dict.keys()]
    av_iter_times = {
        k1: np.mean(
            [timings_dict[k1][k2]["iteration_time"] for k2 in timings_dict[k1].keys()]
        )
        for k1 in timings_dict.keys()
    }

    fig, ax = plt.subplots(figsize=(10, 5))

    # Sort values in ascending order of iteration time
    sorted_av_iter = sorted(av_iter_times.items(), key=lambda x: x[1])
    sorted_av_iter_vals = [sorted_av_iter[val][1] for val in range(len(sorted_av_iter))]

    sorted_ks = [k for k, v in sorted_av_iter]
    xs = np.arange(len(ks))  # Label locations

    # slightly hacky way to get rasterio resampling information
    if isinstance(ks[0], enums.Resampling):
        ks = [resampling.name for resampling in ks]

    # Generate colors for the bar plot
    colors = sns.color_palette(
        get_n_colors_from_hexes(len(sorted_ks), ColourMapGenerator().sequential_hexes)
    )

    # Plotting the total duration
    ax.set_xlabel(k1_name)
    ax.set_ylabel("Average iteration duration (s)")
    sns.barplot(sorted_av_iter_vals, palette=colors, ax=ax)
    ax.grid(axis="y")
    ax.set_xticks(xs)
    ax.set_xticklabels(ks)
    fig.tight_layout()  # Adjust layout to make room for both y-axes
    plt.title(title)


def plot_comparative_histograms_visuals(
    arrays,
    labels,
    val_lims: tuple[float, float] = [-100, 100],
    figsize: tuple[float, float] = None,
    cbar_dict=None,
    hist_dict={"bins": 100, "density": False, "alpha": 0.5, "yscale": "log", "range": None},
    n_hist_bins: int = 100,
    map_extents: list[float] = [140, 145, -15, -10],

    # combined: bool = False,   # TODO: allow plotting on single figure
):
    for a_i, array in tqdm(enumerate(arrays), total=len(arrays)):
        fig = plt.figure(figsize=figsize if figsize else (15, 5))
        gs = fig.add_gridspec(1, 2, height_ratios=[1])
        ax_map = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
        ax_hist = fig.add_subplot(gs[0, 1])

        # Create spatial plot with PlateCarree projection
        plot_spatial(
            array,
            fax=(fig, ax_map),
            cbar_dict=cbar_dict if cbar_dict else {"orientation": "horizontal"},
            val_lims=val_lims,
        )

        if map_extents:
            ax_map.set_extent(map_extents, crs=ccrs.PlateCarree())

        # Create histogram plot
        ax_hist.hist(
            array.values.flatten(),
            bins=hist_dict["bins"],
            alpha=hist_dict["alpha"],
            label=labels[a_i],
            density=hist_dict["density"],
        )
        ax_hist.set_title(labels[a_i])
        if hist_dict["yscale"] == "log":
            ax_hist.set_yscale("log")

        if hist_dict["range"]:
            ax_hist.set_xlim(hist_dict["range"])

        # Add labels
        ax_hist.set_xlabel("Value")
        if hist_dict["density"]:
            ax_hist.set_ylabel("Density")
        else:
            ax_hist.set_ylabel("Frequency")


def plot_two_methods_comparative_histograms_visuals(
    arrays1,
    arrays2,
    ax_labels,
    arrays1_label=None,
    arrays2_label=None,
    cbar_dict: dict = None,
    hist_dict={"bins": 100, "density": False, "alpha": 0.5, "yscale": "log", "range": None},
    val_lims: tuple[float, float] = [-100, 100],
    map_extents: list[float] = [140, 145, -15, -10],
):
    if len(arrays1) != len(arrays2):
        raise ValueError("Array of arrays must be the same length")
    for a_i in tqdm(range(len(arrays1)), total=len(arrays1)):
        fig = plt.figure(figsize=(15, 5))
        gs = fig.add_gridspec(1, 3, height_ratios=[1])
        ax_map1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
        ax_hist = fig.add_subplot(gs[0, 1])
        ax_map2 = fig.add_subplot(gs[0, 2], projection=ccrs.PlateCarree())

        # Create spatial plot with PlateCarree projection
        plot_spatial(
            arrays1[a_i],
            fax=(fig, ax_map1),
            cbar_dict=cbar_dict if cbar_dict else {"orientation": "horizontal"},
            val_lims=val_lims,
            title=arrays1_label,
        )
        plot_spatial(
            arrays2[a_i],
            fax=(fig, ax_map2),
            cbar_dict=cbar_dict if cbar_dict else {"orientation": "horizontal"},
            val_lims=val_lims,
            title=arrays2_label,
        )
        if map_extents:
            [
                ax_map.set_extent(map_extents, crs=ccrs.PlateCarree())
                for ax_map in [ax_map1, ax_map2]
            ]

        # Create histogram plot
        ax_hist.hist(
            arrays1[a_i].values.flatten(),
            bins=hist_dict["bins"],
            alpha=hist_dict["alpha"],
            label="first_method" if not arrays1_label else arrays1_label,
            density=hist_dict["density"],
            color="#d83c04",
        )
        ax_hist.hist(
            arrays2[a_i].values.flatten(),
            bins=hist_dict["bins"],
            alpha=hist_dict["alpha"]/2,
            label="second_method" if not arrays2_label else arrays2_label,
            density=hist_dict["density"],
            color="#3B9AB2",
        )

        # selective formatting
        ax_hist.set_title(ax_labels[a_i])
        if hist_dict["yscale"] == "log":
            ax_hist.set_yscale("log")
        if hist_dict["range"]:
            ax_hist.set_xlim(hist_dict["range"])
        ax_hist.set_xlabel("Value")
        if hist_dict["density"]:
            ax_hist.set_ylabel("Density")
        else:
            ax_hist.set_ylabel("Frequency")
        ax_hist.legend()


def grid_subplots(total, wrap=None, **kwargs):
    if wrap is not None:
        cols = min(total, wrap)
        rows = 1 + (total - 1) // wrap
    else:
        cols = total
        rows = 1
    fig, ax = plt.subplots(rows, cols, **kwargs)
    return fig, ax

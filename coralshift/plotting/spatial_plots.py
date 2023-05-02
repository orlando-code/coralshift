import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import xarray as xa
import cartopy.crs as ccrs


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

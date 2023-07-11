import seaborn as sns
import xarray as xa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import PercentFormatter
import matplotlib.colors as mcolors
import matplotlib.cm as mcm
from matplotlib import gridspec
import pandas as pd

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import sklearn.metrics as sklmetrics

from coralshift.plotting import spatial_plots
from coralshift.machine_learning import baselines
from coralshift.utils import utils


def spatial_confusion_matrix_da(
    predicted: xa.DataArray, ground_truth: xa.DataArray, threshold: float = 0.25
) -> xa.DataArray:
    """Compute a spatial confusion matrix based on the predicted and ground truth xa.DataArray.

    Parameters
    ----------
    predicted (xa.DataArray): Predicted values.
    ground_truth (xa.DataArray): Ground truth values.

    Returns
    -------
    tuple[xa.DataArray, dict]: Spatial confusion matrix and dictionary of integer: description pairs

    Notes
    -----
    The spatial confusion matrix assigns the following categories to each cell:
        - No Data: 0
        - True Positives: 1
        - True Negatives: 2
        - False Positives: 3
        - False Negatives: 4
    """

    # compare ground truth and predicted values
    true_positives = xa.where((predicted == 1) & (ground_truth == 1), 1, 0)
    true_negatives = xa.where((predicted == 0) & (ground_truth == 0), 2, 0)
    false_positives = xa.where((predicted == 1) & (ground_truth == 0), 3, 0)
    false_negatives = xa.where((predicted == 0) & (ground_truth == 1), 4, 0)

    category_variable = (
        true_positives + true_negatives + false_positives + false_negatives
    )

    vals_dict = {
        "No Data": 0,
        "True Positives": 1,
        "True Negatives": 2,
        "False Positives": 3,
        "False Negatives": 4,
    }

    return category_variable, vals_dict


def generate_confusion_ds(xa_ds: xa.Dataset, ground_truth_var: str, predicted_var: str):
    """
    Generates confusion matrix values and adds them to the xarray Dataset.

    Parameters
    ----------
        xa_ds (xa.Dataset): Input xarray Dataset.
        ground_truth_var (str): Variable name for ground truth values in the Dataset.
        predicted_var (str): Variable name for predicted values in the Dataset.

    Returns
    -------
        Tuple[xa.Dataset, np.ndarray, Dict]: A tuple containing the modified xarray Dataset, confusion matrix values,
        and a dictionary of value mappings.
    """
    confusion_values, vals_dict = spatial_confusion_matrix_da(
        xa_ds[predicted_var], xa_ds[ground_truth_var]
    )
    xa_ds["comparison"] = confusion_values

    return xa_ds, confusion_values, vals_dict


def plot_spatial_confusion(
    xa_ds: xa.Dataset,
    ground_truth_var: str,
    predicted_var: str,
    vals_dict: dict,
    fax=None,
    cbar_pad=0.1,
) -> xa.Dataset:
    """Plot a spatial confusion matrix based on the predicted and ground truth variables in the xarray dataset.

    Parameters
    ----------
        xa_ds (xarray.Dataset): Input xarray dataset.
        ground_truth_var (str): Name of the ground truth variable in the dataset.
        predicted_var (str): Name of the predicted variable in the dataset.

    Returns
    -------    -------
        xarray.Dataset: Updated xarray dataset with the "comparison" variable added.
    """
    # calculate spatial confusion values and assign to new variable in Dataset
    map_proj = ccrs.PlateCarree()
    if not fax:
        # may need to change this
        fig = plt.figure(figsize=[16, 8])
        ax = plt.axes(projection=map_proj)
    else:
        fig = fax[0]
        ax = fax[1]

    # from Wes Anderson: https://github.com/karthik/wesanderson/blob/master/R/colors.R
    cmap_colors = ["#EEEEEE", "#3B9AB2", "#78B7C5", "#d83c04", "#E1AF00"]

    im = xa_ds["comparison"].plot.imshow(
        ax=ax, vmin=0, vmax=4, cmap=mcolors.ListedColormap(cmap_colors)
    )

    spatial_plots.format_spatial_plot(
        im, fig, ax, title="", name="", cbar=False, edgecolor="black"
    )
    ax.set_aspect("equal")
    # remove old colorbar
    cb = im.colorbar
    cb.remove()

    colorbar = plt.colorbar(im, ax=[ax], location="right", pad=cbar_pad)
    num_ticks = len(cmap_colors)
    vmin, vmax = colorbar.vmin, colorbar.vmax
    colorbar.set_ticks(
        [vmin + (vmax - vmin) / num_ticks * (0.5 + i) for i in range(num_ticks)]
    )
    colorbar.set_ticklabels(list(vals_dict.keys()), rotation=0)


def format_roc(ax: Axes, title: str = "Receiver Operating Characteristic (ROC) Curve"):
    """
    Format the ROC plot axes.

    Parameters
    ----------
        ax (Axes): The matplotlib axes object.
        title (optional): The title of the ROC plot. Defaults to "Receiver Operating Characteristic (ROC) Curve".

    Returns
    -------
        None
    """
    ax.set_title(title)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_aspect("equal", "box")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # plot "random" line for comparison (y=x)
    ax.plot([0, 1], [0, 1], color="gray", linestyle=(5, (10, 3)))
    # Plot the grid
    ax.grid(color="lightgray", linestyle=":", linewidth=0.5)
    ax.set_xticks(np.arange(0, 1.2, 0.2))
    ax.set_yticks(np.arange(0, 1.2, 0.2))


def visualise_region_class_imbalance(region_imbalance_dict: dict) -> None:
    """
    Visualize class imbalance for different regions.

    Parameters
    ----------
        region_imbalance_dict (dict): Dictionary containing class imbalance information for each region.
            The keys represent the region names, and the values are tuples containing the following:
            - Total Grid Cells (int): Total number of grid cells in the region.
            - Fractional Coral Presence (float): Fraction of grid cells containing coral.

    Returns
    -------
        None
    """

    df = pd.DataFrame.from_dict(
        region_imbalance_dict,
        orient="index",
        columns=["Total Grid Cells", "Fractional Coral Presence"],
    )
    df = df.reset_index().rename(columns={"index": "Region"})

    xs, y1s, y2s = df.columns[0], df.columns[1], df.columns[2]

    colors = ["#78B7C5", "#E1AF00"]
    plt.figure(figsize=(10, 6))

    ax = sns.barplot(
        x=xs, y=y1s, data=df, color=colors[0], label="total grid cells containing coral"
    )
    ax2 = ax.twinx()
    sns.barplot(
        x=xs,
        y=y2s,
        data=df,
        color=colors[1],
        label="fraction of grid cells containing coral",
    )

    # format bar widths
    width_scale = 0.45
    for bar in ax.containers[0]:
        bar.set_width(bar.get_width() * width_scale)
    for bar in ax2.containers[0]:
        x = bar.get_x()
        w = bar.get_width()
        bar.set_x(x + w * (1 - width_scale))
        bar.set_width(w * width_scale)

    # format legend
    (h1, l1), (h2, l2) = ax.get_legend_handles_labels(), ax2.get_legend_handles_labels()
    ax.legend(h1, l1, loc="upper left")
    ax2.legend(h2, l2, loc="upper right")

    # formatting
    ax.set_xlabel("GBR Region")
    ax.set_ylabel("Total Grid Cells")
    ax2.set_ylabel("Coral Presence")
    ax2.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylim((0, 4500))
    ax2.set_ylim((0, 0.2))


def count_nonzero_values(xa_da: xa.DataArray) -> int:
    """
    Count the number of nonzero values in a DataArray.

    Parameters
    ----------
        xa_da (xa.DataArray): DataArray containing values.

    Returns
    -------
        int: Number of nonzero values.
    """
    non_zero_count = xa_da.where(xa_da != 0).count().item()
    return non_zero_count


def calculate_total_class_imbalance(
    xa_ds: list[xa.Dataset | xa.DataArray], var_name: str = None
) -> tuple[int, float]:
    """
    Calculate the total class imbalance for multiple datasets or data arrays.

    Parameters
    ----------
        xa_ds (List[Union[xa.Dataset, xa.DataArray]]): List of xarray Datasets or DataArrays.
        var_name (str, optional): Variable name to extract from the datasets. Defaults to None.

    Returns
    -------
        tuple[int, float]: A tuple containing the following:
            - Total number of nonzero values across all datasets.
            - Fraction representing the class imbalance.
    """
    non_zero_vals, total_spatial_vals = [], []
    for xa_d in xa_ds:
        if var_name:
            xa_d = xa_d[var_name]
        non_zero_vals.append(count_nonzero_values(xa_d["gt"]))
        total_spatial_vals.append(
            len(xa_d.latitude.values) * len(xa_d.longitude.values)
        )

    total_non_zero = sum(non_zero_vals)
    fraction = total_non_zero / sum(total_spatial_vals)
    return total_non_zero, fraction


def calculate_region_class_imbalance(xa_ds_dict: dict) -> dict:
    """
    Calculate the class imbalance for each region in a dictionary of xarray datasets.

    Parameters
    ----------
        xa_ds_dict (dict): Dictionary containing xarray datasets or data arrays.
            The keys represent the region names, and the values are xarray datasets or data arrays.

    Returns
    -------
        dict: Dictionary containing class imbalance information for each region.
            The keys represent the region names, and the values are tuples containing the following:
            - Total number of nonzero values across the region.
            - Fraction representing the class imbalance for the region.
    """
    imbalance_dict = {}
    for k, v in xa_ds_dict.items():
        imbalance_dict[k] = calculate_total_class_imbalance([v])
    return imbalance_dict


def plot_spatial_diffs(
    xa_d_pred: xa.DataArray,
    xa_d_gt: xa.DataArray,
    figsize: tuple[float, float] = (16, 14),
    cbar_pad: float = 0.1,
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
    spatial_plots.plot_spatial(
        fax=(fig, ax_r),
        xa_da=xa_diff,
        cmap_type="div",
        symmetric=True,
        cbar_pad=cbar_pad,
    )

    # right plots
    ax_l_t = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
    spatial_plots.plot_spatial(
        xa_d_gt, fax=(fig, ax_l_t), title="ground truth", cbar_pad=cbar_pad
    )
    ax_l_b = fig.add_subplot(gs[1, 1], projection=ccrs.PlateCarree())
    spatial_plots.plot_spatial(
        xa_d_pred, fax=(fig, ax_l_b), title="inferred label", cbar_pad=cbar_pad
    )


def plot_confusion_matrix(
    labels, predictions, label_threshold: float = 0, fax=None, colorbar: bool = False
) -> None:
    """
    Plot the confusion matrix.

    Parameters
    ----------
        labels (Any): True labels for comparison.
        predictions (Any): Predicted labels.
        label_threshold (float, optional): Label threshold. Defaults to 0.
        fax (Optional[Any], optional): Axes to plot the confusion matrix. Defaults to None.
        colorbar (bool, optional): Whether to show the colorbar. Defaults to False.

    Returns
    -------
        None
    """
    cmap = spatial_plots.get_cbar("lim")

    if not utils.check_discrete(predictions):
        labels = baselines.threshold_array(predictions, threshold=label_threshold)
        predictions = baselines.threshold_array(predictions, threshold=label_threshold)

    classes = ["coral absent", "coral present"]
    # initialise confusion matrix
    cm = sklmetrics.confusion_matrix(labels, predictions)
    disp = sklmetrics.ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=classes
    )
    if fax:
        disp.plot(cmap=cmap, colorbar=colorbar, text_kw={"color": "black"}, ax=fax[1])
    else:
        disp.plot(cmap=cmap, colorbar=colorbar, text_kw={"color": "black"})


def model_output_to_spatial_confusion(
    label: pd.Series | np.ndarray,
    prediction: pd.Series | np.ndarray,
    threshold: float = 0.25,
    lat_lims: tuple[float] = None,
    lon_lims: tuple[float] = None,
    fax=None,
    cbar_pad=0.1,
) -> None:
    """
    Generates a spatial confusion matrix plot based on label and prediction arrays.

    Parameters
    ----------
        label (pd.Series or np.ndarray): True labels.
        prediction (pd.Series or np.ndarray): Predicted labels.
        threshold (float, optional): Threshold value for discretizing the prediction and label arrays. Defaults to 0.25.
        lat_lims (tuple[float], optional): Latitude limits for defining a spatial region. Defaults to None.
        lon_lims (tuple[float], optional): Longitude limits for defining a spatial region. Defaults to None.
        fax (object, optional): Fax object for plotting. Defaults to None.
        cbar_pad (float, optional): Padding between the colorbar and the plot. Defaults to 0.1.

    Returns
    -------
        None
    """
    if not utils.check_discrete(prediction) or not utils.check_discrete(label):
        prediction = baselines.threshold_array(prediction, threshold=threshold)
        label = baselines.threshold_array(label, threshold=threshold)

    ds = baselines.outputs_to_xa_ds(label, prediction)
    confusion_values, vals_dict = spatial_confusion_matrix_da(
        ds["predictions"], ds["labels"]
    )
    if lat_lims and lon_lims:
        region = {
            "latitude": slice(min(lat_lims), max(lat_lims)),
            "longitude": slice(min(lon_lims), max(lon_lims)),
        }
        ds = confusion_values.sel(region)
    ds["comparison"] = confusion_values

    plot_spatial_confusion(
        ds, "labels", "predictions", vals_dict=vals_dict, fax=fax, cbar_pad=cbar_pad
    )


def plot_pixelwise_model_result(
    all_data,
    labels,
    predictions,
    model_type: str = " ",
    figsize=[20, 20],
    thresh_val=-0.25,
    lat_lims=None,
    lon_lims=None,
):
    """
    Plot the pixel-wise model result.

    Parameters
    ----------
        all_data (Any): Data for plotting (gt values).
        labels (Any): Labels for comparison.
        predictions (Any): Model predictions.
        model_type (str, optional): Model type. Defaults to " ".
        figsize (List[int], optional): Figure size. Defaults to [20, 20].
        thresh_val (float, optional): Threshold value. Defaults to -0.25.
        lat_lims (List[float], optional): Latitude limits. Defaults to None.
        lon_lims (List[float], optional): Longitude limits. Defaults to None.

    Returns
    -------
        None
    """
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 2)

    # left plot (total map)
    # ax_context = fig.add_subplot(gs[:, 0], projection=ccrs.PlateCarree())
    # spatial_plots.plot_spatial(fax=(fig, ax_context), xa_da=all_data["gt"], cbar=False)
    if model_type == "brt" or model_type == "rf_reg":
        predictions = baselines.threshold_array(predictions, thresh_val)

    # right plots (spatial cm, cm)
    ax_spatial_cm = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    model_output_to_spatial_confusion(
        labels,
        predictions,
        lat_lims=lat_lims,
        lon_lims=lon_lims,
        fax=(fig, ax_spatial_cm),
        cbar_pad=0.11,
    )


def plot_train_test_spatial(
    xa_da: xa.DataArray,
    figsize: tuple[float, float] = (7, 7),
    bath_mask: xa.DataArray = None,
):
    """
    Plot two spatial variables from a dataset with different colors and labels.

    Parameters
    ----------
    dataset (xarray.Dataset): The dataset containing the variables.

    Returns
    -------
    None
    """
    # Create a figure and axes
    fig, ax = plt.subplots(
        figsize=figsize, subplot_kw=dict(projection=ccrs.PlateCarree())
    )

    cmap = spatial_plots.get_cbar()
    bounds = [0, 0.5, 1]
    if bath_mask.any():
        xa_da = xa_da.where(bath_mask, np.nan)

    _ = xa_da.isel(time=-1).plot.pcolormesh(ax=ax, cmap=cmap, add_colorbar=False)
    ax.set_aspect("equal")
    ax.set_title("Geographical visualisation of train-test split")
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            "physical", "land", "10m", edgecolor="black", facecolor="#cccccc"
        )
    )
    ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
    # plt.colorbar(im)
    # format categorical colorbar
    bounds = [0, 0.5, 1]
    norm = mcolors.BoundaryNorm(bounds, 2)

    # calculate the position of the tick labels
    # min_, max_ = 0, 1
    positions = [0.25, 0.75]
    val_lookup = dict(zip(positions, ["train", "test"]))

    def formatter_func(x, pos):
        "The two parameters are the value and tick position"
        val = val_lookup[x]
        return str(val)

    formatter = plt.FuncFormatter(formatter_func)
    fig.colorbar(
        ax=ax,
        mappable=mcm.ScalarMappable(norm=norm, cmap=cmap),
        ticks=positions,
        format=formatter,
        spacing="proportional",
        pad=0.1,
        fraction=0.046,
    )
    return xa_da


############
# DEPRECATED
############

# def plot_spatial_confusion(
#     xa_ds: xa.Dataset, ground_truth_var: str, predicted_var: str
# ) -> xa.Dataset:
#     """Plot a spatial confusion matrix based on the predicted and ground truth variables in the xarray dataset.

#     Parameters
#     ----------
#         xa_ds (xarray.Dataset): Input xarray dataset.
#         ground_truth_var (str): Name of the ground truth variable in the dataset.
#         predicted_var (str): Name of the predicted variable in the dataset.

#     Returns
#     -------    -------
#         xarray.Dataset: Updated xarray dataset with the "comparison" variable added.
#     """
#     # calculate spatial confusion values and assign to new variable in Dataset
#     xa_ds, confusion_values, vals_dict = generate_confusion_ds(
#         xa_ds, ground_truth_var=ground_truth_var, predicted_var=predicted_var
#     )

#     # from Wes Anderson: https://github.com/karthik/wesanderson/blob/master/R/colors.R
#     cmap = ["#EEEEEE", "#3B9AB2", "#78B7C5", "#F21A00", "#E1AF00"]
#     ax = sns.heatmap(confusion_values, cmap=cmap, vmin=0, vmax=5)
#     ax.set_aspect("equal")
#     # format colourbar
#     colorbar = ax.collections[0].colorbar
#     num_ticks = len(cmap)
#     vmin, vmax = colorbar.vmin, colorbar.vmax
#     colorbar.set_ticks(
#         [vmin + (vmax - vmin) / num_ticks * (0.5 + i) for i in range(num_ticks)]
#     )
#     colorbar.set_ticklabels(list(vals_dict.keys()))

import seaborn as sns
import xarray as xa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import PercentFormatter
import pandas as pd


def spatial_confusion_matrix_da(
    predicted: xa.DataArray, ground_truth: xa.DataArray
) -> xa.DataArray:
    """Compute a spatial confusion matrix based on the predicted and ground truth xa.DataArray.

    Parameters
    ----------
    predicted (xa.DataArray): Predicted values.
    ground_truth (xa.DataArray): Ground truth values.

    Returns
    -------    -------
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


def plot_spatial_confusion(
    xa_ds: xa.Dataset, ground_truth_var: str, predicted_var: str
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
    confusion_values, vals_dict = spatial_confusion_matrix_da(
        xa_ds[predicted_var], xa_ds[ground_truth_var]
    )
    xa_ds["comparison"] = confusion_values

    # from Wes Anderson: https://github.com/karthik/wesanderson/blob/master/R/colors.R
    cmap = ["#EEEEEE", "#3B9AB2", "#78B7C5", "#F21A00", "#E1AF00"]
    ax = sns.heatmap(confusion_values, cmap=cmap, vmin=0, vmax=5)

    # format colourbar
    colorbar = ax.collections[0].colorbar
    num_ticks = len(cmap)
    vmin, vmax = colorbar.vmin, colorbar.vmax
    colorbar.set_ticks(
        [vmin + (vmax - vmin) / num_ticks * (0.5 + i) for i in range(num_ticks)]
    )
    colorbar.set_ticklabels(list(vals_dict.keys()))


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


def visualise_region_class_imbalance(region_imbalance_dict: dict):
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

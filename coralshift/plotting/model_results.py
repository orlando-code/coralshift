import seaborn as sns
import xarray as xa
import numpy as np
from matplotlib.axes import Axes


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

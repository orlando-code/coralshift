import seaborn as sns
import xarray as xa
import matplotlib as plt
from matplotlib.axes import Axes
import numpy as np
import sklearn.metrics as sklmetrics
import pandas as pd
import pickle
from pathlib import Path
from coralshift.plotting import spatial_plots


def spatial_confusion_matrix_da(
    predicted: xa.DataArray, ground_truth: xa.DataArray
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
    -------
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


def investigate_label_thresholds(
    thresholds: list[float],
    y_test: np.ndarray | pd.Series,
    y_predictions: np.ndarray | pd.Series,
    figsize=[7, 7],
):
    """Plot ROC curves with multiple lines for different label thresholds.

    Parameters
    ----------
        thresholds (list[float]): List of label thresholds.
        y_test (np.ndarray or pd.Series): True labels.
        y_predictions (np.ndarray or pd.Series): Predicted labels.
        figsize (list, optional): Figure size for the plot. Default is [7, 7].

    Returns
    -------
        None
    """
    f, ax = plt.subplots(figsize=figsize)
    # prepare colour assignment
    color_map = spatial_plots.get_cbar("seq")
    num_colors = len(thresholds)
    colors = [color_map(i / num_colors) for i in range(num_colors)]

    # plot ROC curves
    for c, thresh in enumerate(thresholds):
        binary_y_labels, binary_predictions = threshold_label(
            y_test, y_predictions, thresh
        )
        fpr, tpr, _ = sklmetrics.roc_curve(
            binary_y_labels, binary_predictions, drop_intermediate=False
        )
        roc_auc = sklmetrics.auc(fpr, tpr)

        label = f"{thresh:.01f} | {roc_auc:.02f}"
        ax.plot(fpr, tpr, label=label, color=colors[c])

    # format
    format_roc(
        ax=ax,
        title="Receiver Operating Characteristic (ROC) Curve\nfor several coral presence/absence thresholds",
    )
    ax.legend(title="threshold value | auc")


def format_roc(ax=Axes, title: str = "Receiver Operating Characteristic (ROC) Curve"):
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_aspect("square")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])


def evaluate_model(y_test: np.ndarray | pd.Series, predictions: np.ndarray):
    """
    Evaluate a model's performance using regression and classification metrics.

    Parameters
    ----------
        y_test (np.ndarray or pd.Series): True labels.
        predictions (np.ndarray or pd.Series): Predicted labels.

    Returns
    -------
        tuple[float, float]: Tuple containing the mean squared error (regression metric) and binary cross-entropy
            (classification metric).
    """
    # calculate regression (mean-squared error) metric
    mse = sklmetrics.mean_squared_error(y_test, predictions)

    # calculate classification (binary cross-entropy/log_loss) metric
    y_thresh, y_pred_thresh = threshold_label(y_test, predictions)
    bce = sklmetrics.log_loss(y_thresh, y_pred_thresh)

    return mse, bce


def threshold_label(
    labels: np.ndarray | pd.Series,
    predictions: np.ndarray | pd.Series,
    threshold: float,
) -> tuple[np.ndarray]:
    """Apply thresholding to labels and predictions.

    Parameters
    ----------
        labels (np.ndarray or pd.Series): True labels.
        predictions (np.ndarray or pd.Series): Predicted labels.
        threshold (float): Threshold value for binary classification.

    Returns
    -------
        tuple[np.ndarray]: Tuple containing thresholded labels and thresholded predictions.
    """
    thresholded_labels = np.where(np.array(labels) > threshold, 1, 0)
    thresholded_preds = np.where(np.array(predictions) > threshold, 1, 0)
    return thresholded_labels, thresholded_preds


def save_sklearn_model(model, savedir: Path | str, filename: str) -> None:
    """
    Save a scikit-learn model to a file using pickle.

    Parameters
    ----------
    model : object
        The scikit-learn model object to be saved.
    savedir : Union[pathlib.Path, str]
        The directory path where the model file should be saved.
    filename : str
        The name of the model file.

    Returns
    -------
    None
    """

    save_path = (Path(savedir) / filename).with_suffix(".pickle")

    if not save_path.is_file():
        with open(save_path, "wb") as f:
            pickle.dump(model, f)
        print(f"Saved model to {save_path}.")
    else:
        print(f"{save_path} already exists.")

    return save_path

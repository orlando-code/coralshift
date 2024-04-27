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

# from tqdm.auto import tqdm

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import sklearn.metrics as sklmetrics

from coralshift.plotting import spatial_plots
from coralshift.machine_learning import static_models
from coralshift.utils import utils, file_ops, config
from coralshift.processing import ml_processing

# from coralshift import functions_creche
from coralshift.processing import spatial_data

import xgboost as xgb


class AnalyseResults:
    def __init__(
        self,
        model,
        trains: tuple[pd.DataFrame, pd.DataFrame] = None,
        tests: tuple[pd.DataFrame, pd.DataFrame] = None,
        vals: tuple[pd.DataFrame, pd.DataFrame] = None,
        save_graphs: bool = True,
        config_info: dict = None,
        presentation_format: bool = False,
    ):
        self.model = model
        self.trains = trains
        self.tests = tests
        self.vals = vals
        self.save_graphs = save_graphs
        self.config_info = config_info
        self.ds_type = None
        self.presentation_format = presentation_format

    def make_predictions(self, X):
        num_points = X.shape[0]
        # convert data to DMatrix format if necessary
        if "xgb" in self.config_info["model_code"]:
            X = xgb.DMatrix(X)
            num_points = X.num_row()

        # if self.predictions is None:  # changed to compute predictions anew each time from model
        print(f"\tRunning inference on {num_points} datapoints...")
        return self.model.predict(X)

    def record_metrics(self, y, predictions):
        data_type = static_models.ModelInitializer(
            self.config_info["model_code"]
        ).get_data_type()

        metric_header = f"{self.ds_type}_metrics" if self.ds_type else "metrics"
        metric_info = {f"{metric_header}": {}}
        if data_type == "continuous":
            # regression metrics
            metric_info[metric_header]["r2_score"] = float(
                sklmetrics.r2_score(y, predictions)
            )
            metric_info[metric_header]["mse"] = float(
                sklmetrics.mean_squared_error(y, predictions)
            )
            metric_info[metric_header]["mae"] = float(
                sklmetrics.mean_absolute_error(y, predictions)
            )

        elif data_type == "discrete":
            [y, predictions] = [
                ml_processing.cont_to_class(
                    y, self.config_info["regressor_classification_threshold"]
                ),
                ml_processing.cont_to_class(
                    predictions, self.config_info["regressor_classification_threshold"]
                ),
            ]
            # classification metrics
            metric_info[metric_header]["f1_score"] = float(
                sklmetrics.f1_score(y, predictions)
            )
            metric_info[metric_header]["accuracy"] = sklmetrics.accuracy_score(
                y, predictions
            )
            metric_info[metric_header]["balanced_accuracy"] = float(
                sklmetrics.balanced_accuracy_score(y, predictions)
            )
        else:
            raise ValueError(f"Data type {data_type} not recognised.")

        # write metric info to config file
        config_fp = self.config_info["file_paths"]["config"]
        print(f"Saving {metric_header} to {config_fp}...")
        file_ops.edit_yaml(config_fp, metric_info)

    def get_plot_dir(self):
        runs_dir = config.runs_dir
        plot_dir_id = self.config_info["file_paths"]["config"].split("_CONFIG")[0]
        plot_dir_child = (
            f"{plot_dir_id}/{self.ds_type}" if self.ds_type else plot_dir_id
        )
        plot_dir = runs_dir / "plots" / plot_dir_child
        plot_dir.mkdir(parents=True, exist_ok=True)

        return plot_dir

    def save_fig(self, fn: str):
        if self.save_graphs:
            plot_dir = self.get_plot_dir()
            plt.savefig(plot_dir / f"{fn}.png")

    def plot_confusion_matrix(self, y, predictions):
        plot_confusion_matrix(
            labels=y,
            predictions=predictions,
            label_threshold=self.config_info["regressor_classification_threshold"],
            presentation_format=self.presentation_format,
        )
        self.save_fig(fn="confusion_matrix")

    def plot_regression(self, y, predictions):
        plot_regression_histograms(
            y, predictions, presentation_format=self.presentation_format
        )
        self.save_fig(fn="regression")

    def plot_spatial_inference_comparison(self, y, predictions):
        plot_spatial_inference_comparison(y, predictions, self.presentation_format)
        self.save_fig(fn="spatial_inference_comparison")

    def plot_spatial_confusion_matrix(self, y, predictions):

        confusion_values = plot_spatial_confusion_matrix(
            y,
            predictions,
            self.config_info["regressor_classification_threshold"],
            presentation_format=self.presentation_format,
        )
        self.save_fig(fn="spatial_confusion_matrix")

        return confusion_values

    def produce_metrics(self, y, predictions):
        self.record_metrics(y, predictions)

    def produce_plots(self, y, predictions):
        data_type = static_models.ModelInitializer(
            self.config_info["model_code"]
        ).get_data_type()

        if data_type == "continuous":
            self.plot_regression(y, predictions)
        self.plot_confusion_matrix(y, predictions)
        self.plot_spatial_inference_comparison(y, predictions)
        self.plot_spatial_confusion_matrix(y, predictions)

    def analyse_results(self):
        ds_types = ["trains", "tests", "vals"]
        for i, ds in enumerate([self.trains, self.tests, self.vals]):
            self.ds_type = ds_types[i]
            predictions = self.make_predictions(ds[0])

            self.produce_plots(ds[1], predictions)
            self.produce_metrics(ds[1], predictions)


def generate_spatial_confusion_matrix_da(
    predicted: xa.DataArray, ground_truth: xa.DataArray
) -> xa.DataArray:
    """Compute a spatial confusion matrix based on the predicted and ground truth xa.DataArray.

    Parameters
    ----------
    predicted (xa.DataArray): Predicted values.
    ground_truth (xa.DataArray): Ground truth values.

    Returns
    -------
    xa.DataArray: Spatial confusion matrix

    """
    # compare ground truth and predicted values
    true_positives = xa.where((predicted == 1) & (ground_truth == 1), 1, 0)
    true_negatives = xa.where((predicted == 0) & (ground_truth == 0), 2, 0)
    false_positives = xa.where((predicted == 1) & (ground_truth == 0), 3, 0)
    false_negatives = xa.where((predicted == 0) & (ground_truth == 1), 4, 0)

    category_variable = (
        true_positives + true_negatives + false_positives + false_negatives
    )

    return category_variable


def get_confusion_vals_dict():
    return {
        "No Data": 0,
        "True Positives": 1,
        "True Negatives": 2,
        "False Positives": 3,
        "False Negatives": 4,
    }


# literally generated by running models on a loop in a ntoebook and printing: automate
def plot_regression_histograms(
    actual, predicted, n_bins: int = 20, presentation_format: bool = False
):
    overfit_r2 = sklmetrics.r2_score(actual, predicted)

    fig, ax = plt.subplots(ncols=2, figsize=(10, 4))
    # plot y=x
    ax[0].plot(
        [0, 1],
        [0, 1],
        color="gray",
        linestyle=(5, (10, 3)),
        label=f"Inference R$^2$ = {overfit_r2:.2f}",
    )
    # plot scatter
    ax[0].scatter(actual, predicted, alpha=0.1, color="#d83c04")
    # plot histogram
    ax[1].hist(actual, bins=n_bins, color="#d83c04", alpha=0.5, label="actual")
    ax[1].hist(predicted, bins=n_bins, color="#3B9AB2", alpha=0.5, label="predicted")

    # format
    plt.suptitle("Inferred-actual comparison")
    ax[0].set_ylabel("Predicted coral presence")
    ax[0].set_xlabel("Actual coral presence")
    ax[1].set_xlabel("Actual coral presence")

    if presentation_format:
        [spatial_plots.customize_plot_colors(fig, a) for a in [ax[0], ax[1]]]
        [a.legend(facecolor="#212121", labelcolor="white") for a in [ax[0], ax[1]]]
    else:
        [a.legend() for a in [ax[0], ax[1]]]


def plot_performance_vs_resolution():
    """Ported directly from pipeline.ipynb
    TODO: clean this up and automate. Please."""
    resolutions = [1, 0.5, 0.25, 0.1, 0.05, 0.01]
    man_accuracy_scores = [
        0.98828125,
        0.990234375,
        0.98876953125,
        0.9949609375,
        0.99544921875,
        0.9478665610321997,
    ]
    man_balanced_accuracy_scores = [
        0.5,
        0.5694620030903217,
        0.5537305421363392,
        0.7689423737055896,
        0.8129974262710511,
        0.8566799050622791,
    ]
    man_f1_scores = [
        0.0,
        0.16666666666666666,
        0.17857142857142855,
        0.5956112852664577,
        0.6750348675034867,
        0.7631139110311135,
    ]

    # Colors
    colors = ["#3B9AB2", "#EBCC2A", "#d83c04"]
    scores = [man_accuracy_scores, man_balanced_accuracy_scores, man_f1_scores]
    score_names = ["Accuracy", "Balanced accuracy", "F1 score"]

    # Plotting
    fig, ax = plt.subplots(dpi=300)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    for idx, score in enumerate(scores):
        ax.scatter(
            resolutions[: len(score)],
            score,
            label=score_names[idx],
            color=colors[idx],
            zorder=3,
        )
        ax.plot(resolutions[: len(score)], score, linestyle=":", color=colors[idx])

    for res in resolutions:
        ax.axvline(res, zorder=0, lw=0.5, c="lightgrey")

    ax.axhline(0.6, c=colors[0], label="Literature baseline accuracy")
    # Adding labels and title
    ax.set_xlabel("Resolution (degrees)")
    ax.set_ylabel("Scores")
    # ax.set_title('Performance vs Resolution')

    ax.set_xticks(resolutions)
    ax.set_xticklabels(rotation=45, labels=resolutions)
    # Adding legend
    ax.legend(loc="upper right", facecolor="#212121", labelcolor="white")

    spatial_plots.customize_plot_colors(fig, ax)


def plot_spatial_confusion(
    comparison_xa: xa.DataArray,
    figsize: tuple[float, float] = [16, 12],
    fax: bool = None,
    cbar_pad: float = 0.1,
    presentation_format: bool = True,
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
    if not fax:
        fig, ax = spatial_plots.generate_geo_axis(figsize=figsize)
    else:
        fig, ax = fax[0], fax[1]

    # fetch meaning of values
    vals_dict = get_confusion_vals_dict()
    cmap_colors = ["#EEEEEE", "#3B9AB2", "#cae7ed", "#d83c04", "#E1AF00"]

    im = comparison_xa.plot(
        ax=ax,
        cmap=mcolors.ListedColormap(cmap_colors),
        vmin=0,
        vmax=4,
        add_colorbar=True,
        transform=ccrs.PlateCarree(),
        alpha=1,
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

    if presentation_format:
        label_style_dict = {"fontsize": 12, "color": "white", "rotation": 45}
        spatial_plots.customize_plot_colors(fig, ax)
        cbar_label_color = "white"
    else:
        label_style_dict = None
        cbar_label_color = "black"

    colorbar.set_ticklabels(list(vals_dict.keys()), color=cbar_label_color)

    spatial_plots.format_spatial_plot(
        im,
        fig,
        ax,
        title="",
        cbar_name="",
        cbar=False,
        edgecolor="black",
        label_style_dict=label_style_dict,
        presentation_format=presentation_format,
    )


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


def plot_spatial_confusion_matrix(
    y, predictions, threshold, presentation_format: bool = True
):
    spatial_predictions_ds = spatial_data.spatial_predictions_from_data(y, predictions)

    spatial_predictions_ds = spatial_predictions_ds.sel(
        latitude=slice(-15, -10), longitude=slice(140, 145)
    )

    # Threshold the values while retaining NaNs
    thresholded_predictions_values = ml_processing.cont_to_class(
        spatial_predictions_ds["predictions"].values,
        threshold,
    )
    thresholded_labels_values = ml_processing.cont_to_class(
        spatial_predictions_ds["label"].values,
        threshold,
    )

    # Create a new DataArray with thresholded values
    thresholded_ds = spatial_predictions_ds.copy()
    thresholded_ds["thresholded_labels"] = (
        thresholded_ds["label"].dims,
        thresholded_labels_values,
    )
    thresholded_ds["thresholded_predictions"] = (
        thresholded_ds["predictions"].dims,
        thresholded_predictions_values,
    )

    confusion_values = generate_spatial_confusion_matrix_da(
        predicted=thresholded_ds["thresholded_predictions"],
        ground_truth=thresholded_ds["thresholded_labels"],
    )

    plot_spatial_confusion(confusion_values, presentation_format=presentation_format)

    return confusion_values


def plot_confusion_matrix(
    labels,
    predictions,
    label_threshold: float = 0,
    fax=None,
    colorbar: bool = False,
    presentation_format: bool = False,
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
    cmap = spatial_plots.get_cbar()

    if not utils.check_discrete(predictions) or not utils.check_discrete(labels):
        labels = ml_processing.cont_to_class(labels, threshold=label_threshold)
        predictions = ml_processing.cont_to_classy(
            predictions, threshold=label_threshold
        )

    classes = ["coral absent", "coral present"]
    # initialise confusion matrix
    cm = sklmetrics.confusion_matrix(
        labels, predictions, labels=[0, 1], normalize="true"
    )
    disp = sklmetrics.ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=classes
    )
    if not fax:
        fax = plt.subplots()

    disp.plot(cmap=cmap, colorbar=colorbar, text_kw={"color": "black"}, ax=fax[1])

    if presentation_format:
        spatial_plots.customize_plot_colors(fax[0], fax[1])


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


def plot_spatial_inference_comparison(y, predictions, presentation_format: bool = True):
    spatial_predictions = spatial_data.spatial_predictions_from_data(y, predictions)

    f, ax = plt.subplots(
        ncols=2,
        figsize=(16, 12),
        dpi=300,
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    spatial_plots.plot_spatial(
        spatial_predictions["label"],
        fax=(f, ax[0]),
        cbar_orientation="horizontal",
        cbar=True,
        presentation_format=presentation_format,
    )
    spatial_plots.plot_spatial(
        spatial_predictions["predictions"],
        fax=(f, ax[1]),
        cbar_orientation="horizontal",
        cbar=True,
        presentation_format=presentation_format,
    )
    return f, ax


# from baselines.py May come in useful
####################################################################################################


# def n_random_runs_preds(
#     model,
#     xa_dss: list[xa.Dataset],
#     runs_n: int = 10,
#     data_type: str = "continuous",
#     test_fraction: float = 0.25,
#     split_type: str = "pixel",
#     train_test_lat_divide: int = float,
#     train_direction: str = "N",
#     bath_mask: bool = True,
# ) -> list[tuple[list]]:
#     """
#     Perform multiple random test runs for inference using a model.

#     Parameters
#     ----------
#         model: The model used for inference.
#         runs_n: The number of random test runs.
#         xa_ds: The xarray Dataset containing the data.
#         test_fraction (optional): The fraction of data to use for testing. Defaults to 0.25.
#         bath_mask (optional): Whether to apply a bathymetry mask during splitting. Defaults to True.

#     Returns
#     -------
#         run_outcomes: A list of tuples containing the true labels and predicted values for each test run.
#     """
#     # TODO: allow spatial splitting, perhaps using **kwarg functionality to declare lat/lon limits
#     # prediction_list = []
#     run_outcomes = []
#     for run in tqdm(
#         range(runs_n),
#         desc=f"Inference on {runs_n} train-test splits",
#     ):
#         # select test data
#         _, X_test, _, y_test, _, _ = spatial_split_train_test(
#             xa_dss=xa_dss,
#             data_type=data_type,
#             test_fraction=test_fraction,
#             split_type=split_type,
#             train_test_lat_divide=train_test_lat_divide,
#             train_direction=train_direction,
#             bath_mask=bath_mask,
#         )

#         pred = model.predict(X_test)
#         run_outcomes.append((y_test, pred))

#     return run_outcomes


# def rocs_n_runs(
#     run_outcomes: tuple[list[float]], binarize_threshold: float = 0, figsize=[7, 7]
# ):
#     """
#     Plot ROC curves for multiple random test runs.

#     Parameters
#     ----------
#         run_outcomes: A list of tuples containing the true labels and predicted values for each test run.
#         binarize_threshold (optional): The threshold value for binarizing the labels. Defaults to 0.

#     Returns
#     -------
#         None
#     """
#     # colour formatting
#     color_map = spatial_plots.get_cbar("seq")
#     num_colors = len(run_outcomes)
#     colors = [color_map(i / num_colors) for i in range(num_colors)]

#     f, ax = plt.subplots(figsize=figsize)
#     roc_aucs = []
#     for c, outcome in enumerate(run_outcomes):
#         # cast regression to binary classification for plotting
#         binary_y_labels, binary_predictions = threshold_label(
#             outcome[0], outcome[1], binarize_threshold
#         )

#         fpr, tpr, _ = sklmetrics.roc_curve(
#             binary_y_labels, binary_predictions, drop_intermediate=False
#         )
#         roc_auc = sklmetrics.auc(fpr, tpr)
#         roc_aucs.append(roc_auc)

#         label = f"{roc_auc:.05f}"
#         ax.plot(fpr, tpr, label=label, color=colors[c])

#     # determine minimum and maximumm auc
#     min_auc, max_auc = np.min(roc_aucs), np.max(roc_aucs)
#     # return mean roc
#     mean_auc = np.mean(roc_aucs)

#     # Set legend labels for min and max AUC lines only
#     handles, legend_labels = ax.get_legend_handles_labels()
#     filtered_handles = []
#     filtered_labels = []
#     for handle, label in zip(handles, legend_labels):
#         if label in [f"{min_auc:.05f}", f"{max_auc:.05f}"]:
#             filtered_handles.append(handle)
#             filtered_labels.append(label)

#     if len(filtered_handles) > 1:
#         filtered_handles, filtered_labels = [filtered_handles[0]], [filtered_labels[0]]
#     ax.legend(
#         filtered_handles,
#         filtered_labels,
#         title="Maximum and minimum AUC scores",
#         loc="lower right",
#     )

#     n_runs = len(run_outcomes)
#     # format
#     model_results.format_roc(
#         ax=ax,
#         title=f"""Receiver Operating Characteristic (ROC) Curve\n for {n_runs} randomly initialised test datasets.
#         \nMean AUC {mean_auc:.05f}""",
#     )


# def investigate_label_thresholds(
#     thresholds: list[float],
#     y_test: np.ndarray | pd.Series,
#     y_predictions: np.ndarray | pd.Series,
#     figsize=[7, 7],
# ):
#     """Plot ROC curves with multiple lines for different label thresholds.

#     Parameters
#     ----------
#         thresholds (list[float]): List of label thresholds.
#         y_test (np.ndarray or pd.Series): True labels.
#         y_predictions (np.ndarray or pd.Series): Predicted labels.
#         figsize (list, optional): Figure size for the plot. Default is [7, 7].

#     Returns
#     -------
#         None
#     """
#     f, ax = plt.subplots(figsize=figsize)
#     # prepare colour assignment
#     color_map = spatial_plots.get_cbar("seq")
#     num_colors = len(thresholds)
#     colors = [color_map(i / num_colors) for i in range(num_colors)]

#     # plot ROC curves
#     for c, thresh in enumerate(thresholds):
#         binary_y_labels, binary_predictions = threshold_label(
#             y_test, y_predictions, thresh
#         )
#         fpr, tpr, _ = sklmetrics.roc_curve(
#             binary_y_labels, binary_predictions, drop_intermediate=False
#         )
#         roc_auc = sklmetrics.auc(fpr, tpr)

#         label = f"{thresh:.01f} | {roc_auc:.02f}"
#         ax.plot(fpr, tpr, label=label, color=colors[c])

#     # format
#     model_results.format_roc(
#         ax=ax,
#         title="Receiver Operating Characteristic (ROC) Curve\nfor several coral presence/absence thresholds",
#     )
#     ax.legend(title="threshold value | auc")


# def n_random_runs_preds_across_models(
#     model_types: list[str],
#     models: list,
#     xa_dss: list[xa.Dataset],
#     runs_n: int = 10,
#     test_fraction: float = 0.25,
#     split_type: str = "pixel",
#     train_test_lat_divide: int = float,
#     train_direction: str = "N",
#     bath_mask: bool = True,
# ):
#     model_class = ModelInitializer()
#     outcomes_dict = {}
#     for i, model_type in enumerate(model_types):
#         model = models[i]
#         data_type = model_class.get_data_type(model_type)
#         outcomes = n_random_runs_preds(
#             model=model,
#             xa_dss=xa_dss,
#             runs_n=runs_n,
#             data_type=data_type,
#             test_fraction=test_fraction,
#             split_type=split_type,
#             train_test_lat_divide=train_test_lat_divide,
#             train_direction=train_direction,
#             bath_mask=bath_mask,
#         )
#         outcomes_dict[model_type] = outcomes

#     return outcomes_dict


# def models_rocs_n_runs(
#     model_outcomes: dict[list[float]], binarize_threshold: float = 0, figsize=[7, 7]
# ):
#     """
#     Plot ROC curves for multiple models' outcomes.

#     Parameters
#     ----------
#         model_outcomes: A dictionary where keys are model names and values are lists of outcomes containing the true
#             labels and predicted values for each test run.
#         binarize_threshold (optional): The threshold value for binarizing the labels. Defaults to 0.

#     Returns
#     -------
#         None
#     """
#     # Colour formatting
#     color_map = spatial_plots.get_cbar("seq")
#     num_models = len(model_outcomes)
#     colors = [color_map(i / num_models) for i in range(num_models)]

#     f, ax = plt.subplots(figsize=figsize)
#     roc_aucs = []
#     legend_labels = []

#     for c, (model_name, outcomes) in tqdm(
#         enumerate(model_outcomes.items()),
#         total=len(model_outcomes),
#         desc="Iterating over models",
#     ):
#         roc_aucs = []

#         for i, outcome in enumerate(outcomes):
#             # Cast regression to binary classification for plotting
#             binary_y_labels, binary_predictions = threshold_label(
#                 outcome[0], outcome[1], binarize_threshold
#             )
#             fpr, tpr, _ = sklmetrics.roc_curve(
#                 binary_y_labels, binary_predictions, drop_intermediate=False
#             )
#             roc_auc = sklmetrics.auc(fpr, tpr)
#             roc_aucs.append(roc_auc)

#             # plot final label
#             if i == (len(outcomes) - 2):
#                 # Calculate mean AUC
#                 mean_auc = np.mean(roc_aucs)
#                 # Show label only for the last outcome of each model
#                 label = f"{model_name} | {mean_auc:.03f}"
#                 ax.plot(fpr, tpr, label=label, color=colors[c])
#                 legend_labels.append(label)
#             else:
#                 # Hide label for other outcomes
#                 ax.plot(fpr, tpr, color=colors[c])

#     ax.legend(title="Mean model AUC scores", loc="lower right")
#     # Format the plot
#     n_runs = len(outcomes)
#     format_roc(
#         ax=ax,
#         title=f"Receiver Operating Characteristic (ROC) Curve\n for {n_runs} randomly initialized test datasets.",
#     )


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


# def model_output_to_spatial_confusion(
#     label: pd.Series | np.ndarray,
#     prediction: pd.Series | np.ndarray,
#     threshold: float = 0.25,
#     lat_lims: tuple[float] = None,
#     lon_lims: tuple[float] = None,
#     fax=None,
#     cbar_pad=0.1,
# ) -> None:
#     """
#     Generates a spatial confusion matrix plot based on label and prediction arrays.

#     Parameters
#     ----------
#         label (pd.Series or np.ndarray): True labels.
#         prediction (pd.Series or np.ndarray): Predicted labels.
#         threshold (float, optional): Threshold value for discretizing the prediction and label arrays. Defaults to
# 0.25.
#         lat_lims (tuple[float], optional): Latitude limits for defining a spatial region. Defaults to None.
#         lon_lims (tuple[float], optional): Longitude limits for defining a spatial region. Defaults to None.
#         fax (object, optional): Fax object for plotting. Defaults to None.
#         cbar_pad (float, optional): Padding between the colorbar and the plot. Defaults to 0.1.

#     Returns
#     -------
#         None
#     """
#     if not utils.check_discrete(prediction) or not utils.check_discrete(label):
#         prediction = baselines.threshold_array(prediction, threshold=threshold)
#         label = baselines.threshold_array(label, threshold=threshold)

#     ds = baselines.outputs_to_xa_ds(label, prediction)
#     confusion_values, vals_dict = spatial_confusion_matrix_da(
#         ds["predictions"], ds["labels"]
#     )
#     if lat_lims and lon_lims:
#         region = {
#             "latitude": slice(min(lat_lims), max(lat_lims)),
#             "longitude": slice(min(lon_lims), max(lon_lims)),
#         }
#         ds = confusion_values.sel(region)
#     ds["comparison"] = confusion_values

#     plot_spatial_confusion(
#         ds, "labels", "predictions", vals_dict=vals_dict, fax=fax, cbar_pad=cbar_pad
#     )

# # a spatial confusion matrix by any other name would small so sweet...
# def plot_pixelwise_model_result(
#     all_data,
#     labels,
#     predictions,
#     model_type: str = " ",
#     figsize=[20, 20],
#     thresh_val=-0.25,
#     lat_lims=None,
#     lon_lims=None,
# ):
#     """
#     Plot the pixel-wise model result.

#     Parameters
#     ----------
#         all_data (Any): Data for plotting (gt values).
#         labels (Any): Labels for comparison.
#         predictions (Any): Model predictions.
#         model_type (str, optional): Model type. Defaults to " ".
#         figsize (List[int], optional): Figure size. Defaults to [20, 20].
#         thresh_val (float, optional): Threshold value. Defaults to -0.25.
#         lat_lims (List[float], optional): Latitude limits. Defaults to None.
#         lon_lims (List[float], optional): Longitude limits. Defaults to None.

#     Returns
#     -------
#         None
#     """
#     fig = plt.figure(figsize=figsize)
#     gs = gridspec.GridSpec(2, 2)

#     # left plot (total map)
#     # ax_context = fig.add_subplot(gs[:, 0], projection=ccrs.PlateCarree())
#     # spatial_plots.plot_spatial(fax=(fig, ax_context), xa_da=all_data["gt"], cbar=False)
#     if model_type == "brt" or model_type == "rf_reg":
#         predictions = baselines.threshold_array(predictions, thresh_val)

#     # right plots (spatial cm, cm)
#     ax_spatial_cm = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
#     model_output_to_spatial_confusion(
#         labels,
#         predictions,
#         lat_lims=lat_lims,
#         lon_lims=lon_lims,
#         fax=(fig, ax_spatial_cm),
#         cbar_pad=0.11,
#     )

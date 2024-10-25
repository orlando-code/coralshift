# file ops
from pathlib import Path

# general
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ml
import sklearn.metrics as sklmetrics
from sklearn.inspection import permutation_importance
import xgboost as xgb

# custom
from coralshift.machine_learning import static_models
from coralshift.processing import ml_processing
from coralshift.utils import file_ops
from coralshift.plotting import visualise_results, spatial_plots


class AnalyseResults:
    def __init__(
        self,
        model=None,
        model_code: str = None,
        trains: tuple[pd.DataFrame, pd.DataFrame] = None,
        tests: tuple[pd.DataFrame, pd.DataFrame] = None,
        vals: tuple[pd.DataFrame, pd.DataFrame] = None,
        trains_preds: pd.DataFrame = None,
        tests_preds: pd.DataFrame = None,
        do_plot: bool = False,
        save_graphs: bool = True,
        config_info: dict = None,
        presentation_format: bool = False,
        extent: list[float] = None,
    ):
        self.model = model
        self.model_code = model_code
        self.trains = trains
        self.tests = tests
        self.vals = vals
        self.trains_preds = trains_preds
        self.tests_preds = tests_preds
        self.do_plot = do_plot
        self.save_graphs = save_graphs
        self.config_info = config_info
        self.extent = extent if extent else [*self.config_info["lons"], *self.config_info["lats"]]
        self.ds_type = None
        self.presentation_format = presentation_format

        if config_info:
            self.__dict__.update(config_info)

    def make_predictions(self, X):
        num_points = X.shape[0]
        # convert data to DMatrix format if necessary
        # if "xgb" in self.config_info["model_code"]:
        if (
            "xgb" in self.model_code
        ):  # TODO: check instance of data rather than referring to model_code
            X = xgb.DMatrix(X)
            num_points = X.num_row()

        # if self.predictions is None:  # changed to compute predictions anew each time from model
        print(f"\tRunning inference on {num_points} datapoints...")
        return self.model.predict(X)

    def record_metrics(self, y, predictions):
        data_type = static_models.ModelInitialiser(
            # self.config_info["model_code"]
            self.model_code
        ).get_data_type()  # TODO: look at whether y is discrete or continuous rather than model_code

        metric_header = f"{self.ds_type}_metrics" if self.ds_type else "metrics"
        metric_info = {f"{metric_header}": {}}
        if data_type == "continuous":
            # regression metrics
            metric_info[metric_header]["r2_score"] = round(
                float(sklmetrics.r2_score(y, predictions)), 5
            )
            metric_info[metric_header]["mse"] = round(
                float(sklmetrics.mean_squared_error(y, predictions)), 5
            )
            metric_info[metric_header]["mae"] = round(
                float(sklmetrics.mean_absolute_error(y, predictions)), 5
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
            metric_info[metric_header]["f1_score"] = round(
                float(sklmetrics.f1_score(y, predictions)), 5
            )
            metric_info[metric_header]["accuracy"] = round(
                sklmetrics.accuracy_score(y, predictions), 5
            )
            metric_info[metric_header]["balanced_accuracy"] = round(
                float(sklmetrics.balanced_accuracy_score(y, predictions)), 5
            )

        else:
            raise ValueError(f"Data type {data_type} not recognised.")

        # write metric info to config file
        config_fp = self.config_info["file_paths"]["config"]
        print(f"Saving {metric_header} to {config_fp}...")
        file_ops.edit_yaml(config_fp, metric_info)

    def get_plot_dir(self):
        # runs_dir = config.runs_dir
        # plot_dir_id = self.config_info["file_paths"]["config"].split("_CONFIG")[0]
        fp_root = file_ops.FileHandler(
            config_info=self.config_info, model_code=self.model_code
        ).get_highest_unique_fp_root()
        fp_root_dir = fp_root.parent / fp_root.name

        plot_dir_child = Path(f"{fp_root_dir}/{self.ds_type}")
        # print(plot_dir)
        # plot_dir = runs_dir / "plots" / plot_dir_child
        plot_dir_child.mkdir(parents=True, exist_ok=True)

        return plot_dir_child

    def save_fig(self, fn: str, dpi: int = 300):
        if self.save_graphs:
            plot_dir = self.get_plot_dir()
            plt.savefig(plot_dir / f"{fn}.png")

    def plot_confusion_matrix(self, y, predictions):
        visualise_results.plot_confusion_matrix(
            labels=y,
            predictions=predictions,
            label_threshold=self.config_info["regressor_classification_threshold"],
            presentation_format=self.presentation_format,
        )
        self.save_fig(fn="confusion_matrix")

        if not self.do_plot:
            plt.close()

    def plot_regression(self, y, predictions):
        visualise_results.plot_regression_histograms(
            y, predictions, presentation_format=self.presentation_format
        )
        self.save_fig(fn="regression")

        if not self.do_plot:
            plt.close()

    def plot_spatial_inference_comparison(self, y, predictions):
        f, ax = visualise_results.plot_spatial_inference_comparison(
            y, predictions, self.presentation_format, extent=self.extent
        )
        self.save_fig(fn="spatial_inference_comparison", dpi=f.dpi)

        if not self.do_plot:
            plt.close()

    def plot_spatial_confusion_matrix(self, y, predictions):

        confusion_values, f, ax = visualise_results.plot_spatial_confusion_matrix(
            y,
            predictions,
            self.config_info["regressor_classification_threshold"],
            presentation_format=self.presentation_format,
            extent=self.extent,
        )
        self.save_fig(fn="spatial_confusion_matrix", dpi=f.dpi)

        if not self.do_plot:
            plt.close()

        return confusion_values

    def plot_spatial_residuals(self, y, predictions):
        f, ax = visualise_results.plot_spatial_residuals(
            y,
            predictions,
            extent=self.extent
            # presentation_format=self.presentation_format
        )
        self.save_fig(fn="spatial_differences", dpi=f.dpi)

        if not self.do_plot:
            plt.close()

    def produce_metrics(self, y, predictions):
        self.record_metrics(y, predictions)

    def produce_plots(self, y, predictions):
        data_type = static_models.ModelInitialiser(
            # self.config_info["model_code"]
            self.model_code
        ).get_data_type()  # TODO: look at whether y is discrete or continuous rather than model_code

        if data_type == "continuous":
            self.plot_regression(y, predictions)
            self.plot_spatial_residuals(y, predictions)
        self.plot_confusion_matrix(y, predictions)
        self.plot_spatial_inference_comparison(y, predictions)
        self.plot_spatial_confusion_matrix(y, predictions)

    def analyse_results(self):
        ds_types = ["trains", "tests", "vals"]
        n_samples = self.config_info["hyperparameter_search"]["n_samples"]

        for i, ds in enumerate([self.trains, self.tests, self.vals]):
            self.ds_type = ds_types[i]
            if not self.trains_preds:
                predictions = self.make_predictions(ds[0][:n_samples])
            else:
                predictions = self.trains_preds if self.ds_type == "trains" else self.tests_preds

            self.produce_plots(ds[1][:n_samples], predictions)
            self.produce_metrics(ds[1][:n_samples], predictions)


def permutation_feature_importance(
    model,
    val_X: pd.DataFrame,
    val_y: pd.DataFrame,
    n_repeats: int = 5,
    n_jobs: int = 16,
    random_state: int = 42,
):
    print("Running feature importance check...")
    perm_imp_dict = permutation_importance(
        model,
        val_X,
        val_y,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    # return dataframe with columns as headers, results of each run as rows
    return pd.DataFrame(perm_imp_dict.importances.T, columns=val_X.columns)


def plot_forest_feature_importances(
    model, X_train: pd.DataFrame, n_samples: int = 30, figsize: tuple[float] = None
):
    importances = model.feature_importances_
    descending_indices = np.argsort(importances)

    sorted_importances = importances[descending_indices]
    sorted_labels = X_train.columns[descending_indices]

    # adjust figsize based on number of samples, if not provided
    if not figsize:
        figsize = (10, n_samples * 0.25)

    fig, ax = plt.subplots(figsize=figsize)

    cmap = spatial_plots.ColourMapGenerator().get_cmap("seq")(
        np.linspace(0, 1, n_samples)
    )

    ax.barh(
        range(len(sorted_importances[-n_samples:])),
        sorted_importances[-n_samples:],
        color=cmap,
    )

    # formatting
    ax.set_yticks(range(len(importances[:n_samples])))
    _ = ax.set_yticklabels(sorted_labels[-n_samples:])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.grid(axis="x", linestyle="-", alpha=0.6)
    # plot minor gridlines
    plt.minorticks_on()
    plt.grid(axis="x", which="minor", linestyle="-0", alpha=0.2)


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
#     model_class = ModelInitialiser()
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

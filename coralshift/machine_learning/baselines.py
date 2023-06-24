from __future__ import annotations

import pickle
import json
import time
import xarray as xa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from pathlib import Path
from tqdm import tqdm

from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics as sklmetrics

from coralshift.utils import utils, directories, file_ops
from coralshift.processing import spatial_data
from coralshift.plotting import spatial_plots, model_results


def generate_test_train_coordinates(
    xa_ds: xa.Dataset,
    split_type: str = "pixel",
    test_lats: tuple[float] = None,
    test_lons: tuple[float] = None,
    test_fraction: float = 0.2,
    bath_mask: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate test and train coordinates for a given dataset.

    Parameters
    ----------
        xa_ds (xa.Dataset): The input xarray Dataset.
        split_type (str): The split type, either "pixel" or "region". Default is "pixel".
        test_lats (tuple[float]): The latitude range for the test region. Required for "region" split type.
            Default is None.
        test_lons (tuple[float]): The longitude range for the test region. Required for "region" split type.
            Default is None.
        test_fraction (float): The fraction of data to be used for the test set. Default is 0.2.

    Returns
    -------
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing train coordinates and test coordinates as pandas
            DataFrames.
    """
    # if chosen to restrict to shallow regions only, set all values outside of threshold to nan
    if bath_mask:
        xa_ds = xa_ds.compute()
        bath_mask = spatial_data.generate_var_mask(xa_ds)
        xa_ds = xa_ds.where(bath_mask, np.nan)

    if split_type == "pixel":
        # have to handle time: make sure sampling spatially rather than spatiotempoorally
        train_coordinates, test_coordinates = utils.generate_coordinate_pairs(
            xa_ds, test_fraction
        )

    elif split_type == "region":
        # if specific latitude/longitude boundary not specified, separate region horizontally
        if not (test_lons and test_lats):
            # calculate the number of latitude cells
            num_lats = len(xa_ds.latitude.values)
            # calculate number of latitude rows in test and train sets
            test_size = int(num_lats * test_fraction)
            train_size = num_lats - test_size

            # slice region into test and train xa.Datasets
            train_xa = xa_ds.isel({"latitude": slice(0, train_size)})
            test_xa = xa_ds.isel({"latitude": slice(train_size, num_lats)})

            train_coordinates, _ = utils.generate_coordinate_pairs(train_xa, 0)
            test_coordinates, _ = utils.generate_coordinate_pairs(test_xa, 0)

        # if specific latitude/longitude boundary specified, cut out test region and train on all else
        else:
            test_xa = xa_ds.isel(
                {
                    "latitude": slice(test_lats[0], test_lats[1]),
                    "longitude": slice(test_lons[0], test_lons[1]),
                }
            )
            all_coordinates = utils.generate_coordinate_pairs(xa_ds)

            test_coordinates = utils.generate_coordinate_pairs(test_xa)
            train_coordinates = list(set(all_coordinates - set(test_coordinates)))

    return train_coordinates, test_coordinates


def generate_test_train_coordinates_multiple_areas(
    xa_ds_list: list[xa.Dataset],
    split_type: str = "pixel",
    test_lats: tuple[float] = None,
    test_lons: tuple[float] = None,
    test_fraction: float = 0.2,
    bath_mask: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate test and train coordinates for multiple areas.

    Args:
        xa_ds_list (list[xa.Dataset]): List of xarray datasets.
        split_type (str, optional): Split type. Defaults to "pixel".
        test_lats (tuple[float], optional): Latitude bounds for the test set. Defaults to None.
        test_lons (tuple[float], optional): Longitude bounds for the test set. Defaults to None.
        test_fraction (float, optional): Fraction of data to assign to the test set. Defaults to 0.2.
        bath_mask (bool, optional): Flag to apply a bathymetry mask. Defaults to True.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple of pandas DataFrames containing the train coordinates and test
            coordinates.
    """
    train_coords = pd.concat(
        (
            generate_test_train_coordinates(
                xa_ds, split_type, test_lats, test_lons, test_fraction, bath_mask
            )[0]
            for xa_ds in xa_ds_list
        ),
        ignore_index=True,
        axis=0,
    )
    test_coords = pd.concat(
        (
            generate_test_train_coordinates(
                xa_ds, split_type, test_lats, test_lons, test_fraction, bath_mask
            )[1]
            for xa_ds in xa_ds_list
        ),
        ignore_index=True,
        axis=0,
    )

    return train_coords, test_coords

    # train_coords_dfs, test_coords_dfs = [], []
    # for xa_ds in xa_ds_list:
    #     train_coords, test_coords = generate_test_train_coordinates(
    #         xa_ds=xa_ds,
    #         split_type=split_type,
    #         test_lats=test_lats,
    #         test_lons=test_lons,
    #         test_fraction=test_fraction,
    #         bath_mask=bath_mask,
    #     )
    #     train_coords_dfs.append(train_coords)
    #     test_coords_dfs.append(test_coords)

    # # merge dfs
    # train_coords = pd.concat(train_coords_dfs, ignore_index=True)
    # test_coords = pd.concat(test_coords_dfs, ignore_index=True)


def spatial_split_train_test(
    xa_ds: xa.Dataset,
    gt_label: str = "gt",
    data_type: str = "continuous",
    ignore_vars: list = ["time", "spatial_ref", "band", "depth"],
    split_type: str = "pixel",
    test_lats: tuple[float] = None,
    test_lons: tuple[float] = None,
    test_fraction: float = 0.2,
    bath_mask: bool = True,
) -> tuple:
    """
    Split the input dataset into train and test sets based on spatial coordinates.

    Parameters
    ----------
        xa_ds (xa.Dataset): The input xarray Dataset.
        gt_label: The ground truth label.
        ignore_vars (list): A list of variables to ignore during splitting. Default is
            ["time", "spatial_ref", "band", "depth"].
        split_type (str): The split type, either "pixel" or "region". Default is "pixel".
        test_lats (tuple[float]): The latitude range for the test region. Required for "region" split type.
            Default is None.
        test_lons (tuple[float]): The longitude range for the test region. Required for "region" split type.
            Default is None.
        test_fraction (float): The fraction of data to be used for the test set. Default is 0.2.

    Returns
    -------
        tuple: A tuple containing X_train, X_test, y_train, and y_test.
    """
    # generate lists of tuples specifying coordinates to be used for training and testing
    train_coordinates, test_coordinates = generate_test_train_coordinates(
        xa_ds, split_type, test_lats, test_lons, test_fraction, bath_mask
    )

    # flatten dataset for row indexing and model training
    # compute out dasked chunks, fill Nan values with 0, drop columns which would confuse model
    flattened_data = (
        xa_ds.stack(points=("latitude", "longitude", "time"))
        .compute()
        .to_dataframe()
        .fillna(0)
        .drop(["time", "spatial_ref", "band", "depth"], axis=1)
        .astype("float32")
    )

    # normalise data via min/max scaling
    normalised_data = (flattened_data - flattened_data.min()) / (
        flattened_data.max() - flattened_data.min()
    )

    # return train and test rows from dataframe
    train_rows = utils.select_df_rows_by_coords(normalised_data, train_coordinates)
    test_rows = utils.select_df_rows_by_coords(normalised_data, test_coordinates)

    # assign rows to test and train features/labels
    X_train, X_test = train_rows.drop("gt", axis=1), test_rows.drop("gt", axis=1)
    y_train, y_test = train_rows["gt"], test_rows["gt"]

    if data_type == "discrete":
        y_train, y_test = threshold_array(y_train), threshold_array(y_test)

    return X_train, X_test, y_train, y_test, train_coordinates, test_coordinates


def visualise_train_test_split(xa_ds: xa.Dataset, train_coordinates, test_coordinates):
    """
    Visualizes the training and testing regions on a spatial grid.

    Parameters:
        xa_ds (xarray.Dataset): Input dataset containing spatial grid information.
        train_coordinates (list): List of training coordinates as (latitude, longitude) tuples.
        test_coordinates (list): List of testing coordinates as (latitude, longitude) tuples.

    Returns:
        xarray.DataArray: DataArray with the same coordinates and dimensions as xa_ds, where the spatial pixels
                          corresponding to training and testing regions are color-coded.
    """
    lat_spacing = xa_ds.latitude.values[1] - xa_ds.latitude.values[0]
    lon_spacing = xa_ds.longitude.values[1] - xa_ds.longitude.values[0]

    array_shape = tuple(xa_ds.dims[d] for d in list(xa_ds.dims))
    train_pixs, test_pixs = np.empty(array_shape), np.empty(array_shape)
    train_pixs[:] = np.nan
    test_pixs[:] = np.nan
    # Color the spatial pixels corresponding to training and testing regions
    for train_index in tqdm(train_coordinates, desc="Coloring in training indices..."):
        row, col = spatial_data.find_coord_indices(
            xa_ds, train_index[0], train_index[1], lat_spacing, lon_spacing
        )
        train_pixs[row, col] = 0

    for test_index in tqdm(test_coordinates, desc="Coloring in training indices..."):
        row, col = spatial_data.find_coord_indices(
            xa_ds, test_index[0], test_index[1], lat_spacing, lon_spacing
        )
        test_pixs[row, col] = 1

    train_test_ds = xa.DataArray(
        np.nansum(np.stack((train_pixs, test_pixs)), axis=0),
        coords=xa_ds.coords,
        dims=xa_ds.dims,
    )
    return train_test_ds


def rf_search_grid(
    estimator_lims: tuple[int] = (200, 2000),
    max_features: list[str] = ["auto", "sqrt"],
    max_depth_lims: tuple[int] = (10, 110),
    min_samples_split: list[int] = [2, 5, 10],
    min_samples_leaf: list[int] = [1, 2, 4],
    bootstrap: list[bool] = [True, False],
) -> dict:
    # Number of trees in random forest
    n_estimators = [
        int(x)
        for x in np.linspace(
            start=min(estimator_lims), stop=max(estimator_lims), num=10
        )
    ]
    # Number of features to consider at every split
    max_features = max_features
    # Maximum number of levels in tree
    max_depth = [
        int(x) for x in np.linspace(min(max_depth_lims), max(max_depth_lims), num=11)
    ]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = min_samples_split
    # Minimum number of samples required at each leaf node
    min_samples_leaf = min_samples_leaf
    # Method of selecting samples for training each tree
    bootstrap = bootstrap
    # Create the random grid
    random_grid = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "bootstrap": bootstrap,
    }
    return random_grid


def boosted_regression_search_grid(
    n_estimators_lims: tuple[int] = (100, 2000),
    learning_rate_lims: tuple[float] = (0.001, 1.0),
    max_depth_lims: tuple[int] = (1, 10),
    min_samples_split: list[int] = [2, 5, 10],
    min_samples_leaf: list[int] = [1, 2, 4],
    max_features: list[str] = ["auto", "sqrt"],
    loss: list[str] = ["ls", "lad", "huber", "quantile"],
    subsample_lims: tuple[float] = (0.1, 1.0),
    criterion: list[str] = ["friedman_mse", "mse"],
) -> dict:
    # Number of trees in the ensemble
    n_estimators = [
        int(x)
        for x in np.linspace(
            start=min(n_estimators_lims), stop=max(n_estimators_lims), num=10
        )
    ]
    # Learning rate (shrinkage)
    learning_rate = np.logspace(*np.log10(learning_rate_lims), num=10).tolist()
    # Maximum depth of each tree
    max_depth = [
        int(x)
        for x in np.linspace(
            start=min(max_depth_lims), stop=max(max_depth_lims), num=10
        )
    ]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = min_samples_split
    # Minimum number of samples required at each leaf node
    min_samples_leaf = min_samples_leaf
    # Maximum number of features to consider at each split
    max_features = max_features
    # Loss function to optimize
    loss = loss
    # Fraction of samples to be used for training each tree
    subsample = np.linspace(
        start=subsample_lims[0], stop=subsample_lims[1], num=10
    ).tolist()
    # Splitting criterion
    criterion = criterion

    # Create the random grid
    random_grid = {
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "max_features": max_features,
        "loss": loss,
        "subsample": subsample,
        "criterion": criterion,
    }
    return random_grid


def maximum_entropy_search_grid(
    penalty: list[str] = ["l1", "l2", "elasticnet", "none"],
    dual: list[bool] = [True, False],
    tol: list[float] = [1e-4, 1e-3, 1e-2],
    C: list[float] = [0.1, 1.0, 10.0],
    fit_intercept: list[bool] = [True, False],
    intercept_scaling: list[float] = [1.0, 2.0, 5.0],
    solver: list[str] = ["sag", "saga", "newton-cholesky"],
    max_iter: list[int] = [100, 200, 500],
    multi_class: list[str] = ["auto", "ovr", "multinomial"],
    verbose: list[int] = [0, 1, 2],
    warm_start: list[bool] = [True, False],
) -> dict:
    # Regularization penalty
    penalty = penalty
    # Dual formulation
    dual = dual
    # Convergence tolerance
    tol = tol
    # Inverse of regularization strength
    C = C
    # Fit intercept
    fit_intercept = fit_intercept
    # Intercept scaling
    intercept_scaling = intercept_scaling
    # Solver algorithm
    solver = solver
    # Maximum number of iterations
    max_iter = max_iter
    # Multi-class option
    multi_class = multi_class
    # Verbosity level
    verbose = verbose
    # Warm start
    warm_start = warm_start

    # Create the random grid
    random_grid = {
        "penalty": penalty,
        "dual": dual,
        "tol": tol,
        "C": C,
        "fit_intercept": fit_intercept,
        "intercept_scaling": intercept_scaling,
        "solver": solver,
        "max_iter": max_iter,
        "multi_class": multi_class,
        "verbose": verbose,
        "warm_start": warm_start,
    }

    return random_grid


def n_random_runs_preds(
    model,
    runs_n,
    xa_ds,
    data_type: str = "continuous",
    test_fraction: float = 0.25,
    bath_mask: bool = True,
) -> list[tuple[list]]:
    """
    Perform multiple random test runs for inference using a model.

    Parameters
    ----------
        model: The model used for inference.
        runs_n: The number of random test runs.
        xa_ds: The xarray Dataset containing the data.
        test_fraction (optional): The fraction of data to use for testing. Defaults to 0.25.
        bath_mask (optional): Whether to apply a bathymetry mask during splitting. Defaults to True.

    Returns
    -------
        run_outcomes: A list of tuples containing the true labels and predicted values for each test run.
    """
    # TODO: allow spatial splitting, perhaps using **kwarg functionality to declare lat/lon limits
    # prediction_list = []
    run_outcomes = []
    for run in tqdm(
        range(runs_n),
        desc=f" Running inference on {runs_n} randomly-initialised test splits with {test_fraction} test fraction",
    ):
        # randomly select test data
        _, X_test, _, y_test, _, _ = spatial_split_train_test(
            xa_ds, data_type=data_type, test_fraction=test_fraction, bath_mask=bath_mask
        )

        pred = model.predict(X_test)
        run_outcomes.append((y_test, pred))

    return run_outcomes


def rocs_n_runs(
    run_outcomes: tuple[list[float]], binarize_threshold: float = 0, figsize=[7, 7]
):
    """
    Plot ROC curves for multiple random test runs.

    Parameters
    ----------
        run_outcomes: A list of tuples containing the true labels and predicted values for each test run.
        binarize_threshold (optional): The threshold value for binarizing the labels. Defaults to 0.

    Returns
    -------
        None
    """
    color_map = spatial_plots.get_cbar("seq")
    num_colors = len(run_outcomes)
    colors = [color_map(i / num_colors) for i in range(num_colors)]

    f, ax = plt.subplots(figsize=figsize)
    for c, outcome in enumerate(run_outcomes):
        # cast regression to binary classification for plotting
        binary_y_labels, binary_predictions = model_results.threshold_label(
            outcome[0], outcome[1], binarize_threshold
        )

        fpr, tpr, _ = sklmetrics.roc_curve(
            binary_y_labels, binary_predictions, drop_intermediate=False
        )
        roc_auc = sklmetrics.auc(fpr, tpr)

        label = f"{roc_auc:.02f}"
        ax.plot(fpr, tpr, label=label, color=colors[c])

    n_runs = len(run_outcomes)
    # format
    format_roc(
        ax=ax,
        title=f"Receiver Operating Characteristic (ROC) Curve\n for {n_runs} randomly initialised test datasets.",
    )


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


def threshold_array(array: np.ndarray | pd.Series, threshold: float = 0) -> np.ndarray:
    return np.where(np.array(array) > threshold, 1, 0)


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
    thresholded_labels = threshold_array(labels, threshold)
    thresholded_preds = threshold_array(predictions, threshold)
    return thresholded_labels, thresholded_preds


def calc_time_weighted_mean(xa_da_daily_means: xa.DataArray, period: str):
    """
    Calculates the weighted mean of daily means for a given period.

    Parameters
    ----------
    xa_da_daily_means (xarray.DataArray): The input xarray DataArray of daily means.
    period (str): The time period for grouping (e.g., 'year', 'month', 'week').

    Returns
    -------
    xarray.DataArray: The weighted mean values for the given period.
    """
    group, offset = return_time_grouping_offset(period)
    # determine original crs
    crs = xa_da_daily_means.rio.crs

    # Determine the month length (has no effect on other time periods)
    month_length = xa_da_daily_means.time.dt.days_in_month
    # Calculate the monthly weights
    weights = month_length.groupby(group) / month_length.groupby(group).sum()

    # Setup our masking for nan values
    ones = xa.where(xa_da_daily_means.isnull(), 0.0, 1.0)

    # Calculate the numerator
    xa_da_daily_means_sum = (
        (xa_da_daily_means * weights).resample(time=offset).sum(dim="time")
    )
    # Calculate the denominator
    ones_out = (ones * weights).resample(time=offset).sum(dim="time")

    # weighted average
    return (xa_da_daily_means_sum / ones_out).rio.write_crs(crs, inplace=True)


def calc_timeseries_params(xa_da_daily_means: xa.DataArray, period: str, param: str):
    """
    Calculates time series parameters (mean, standard deviation, min, max) for a given period.

    Parameters
    ----------
    xa_da_daily_means (xarray.DataArray): The input xarray DataArray of daily means.
    period (str): The time period for grouping (e.g., 'year', 'month', 'week').
    param (str): The parameter name.

    Returns
    -------
    xarray.DataArray, xarray.DataArray, tuple(xarray.DataArray, xarray.DataArray): The weighted average,
        standard deviation, and (min, max) values for the given period.
    """
    # determine original crs
    crs = xa_da_daily_means.rio.crs

    base_name = f"{param}_{period}_"
    # weighted average
    weighted_av = calc_time_weighted_mean(xa_da_daily_means, period)

    weighted_av = weighted_av.rename(base_name + "mean")
    # standard deviation of weighted averages
    stdev = (
        weighted_av.std(dim="time", skipna=True)
        .rename(base_name + "std")
        .rio.write_crs(crs, inplace=True)
    )
    # max and min
    min = (
        xa_da_daily_means.min(dim="time", skipna=True)
        .rename(base_name + "min")
        .rio.write_crs(crs, inplace=True)
    )
    max = (
        xa_da_daily_means.max(dim="time", skipna=True)
        .rename(base_name + "max")
        .rio.write_crs(crs, inplace=True)
    )

    # Return the weighted average
    return weighted_av, stdev, (min, max)


def generate_reproducing_metrics(
    resampled_xa_das_dict: dict, target_resolution: float = None
) -> xa.Dataset:
    """
    Generate metrics used in Couce et al (2013, 2023) based on the upsampled xarray DataArrays.

    Parameters
    ----------
        resampled_xa_das_dict (dict): A dictionary containing the upsampled xarray DataArrays.

    Returns
    -------
        xa.Dataset: An xarray Dataset containing the reproduced metrics.
    """
    if target_resolution:
        resolution = target_resolution
    else:
        resolution = np.mean(
            spatial_data.calculate_spatial_resolution(resampled_xa_das_dict["ssr"])
        )

    res_string = f"{resolution:.05f}d"
    filename = utils.replace_dot_with_dash(
        f"{res_string}_arrays/all_{res_string}_comparative"
    )
    save_path = (directories.get_comparison_dir() / filename).with_suffix(".nc")

    if not save_path.is_file():
        # TEMPERATURE
        thetao_daily = resampled_xa_das_dict["thetao"]
        # annual average, stdev of annual averages, annual minimum, annual maximum
        (
            thetao_annual_average,
            _,
            (thetao_annual_min, thetao_annual_max),
        ) = calc_timeseries_params(thetao_daily, "y", "thetao")
        # monthly average, stdev of monthly averages, monthly minimum, monthly maximum
        (
            thetao_monthly_average,
            thetao_monthly_stdev,
            (thetao_monthly_min, thetao_monthly_max),
        ) = calc_timeseries_params(thetao_daily, "m", "thetao")
        # annual range (monthly max - monthly min)
        thetao_annual_range = (thetao_annual_max - thetao_annual_min).rename(
            "thetao_annual_range"
        )
        # weekly minimum, weekly maximum
        _, _, (thetao_weekly_min, thetao_weekly_max) = calc_timeseries_params(
            thetao_daily, "w", "thetao"
        )
        print("Generated thetao data.")

        # SALINITY
        salinity_daily = resampled_xa_das_dict["so"]
        # annual average
        salinity_annual_average, _, _ = calc_timeseries_params(
            salinity_daily, "y", "salinity"
        )
        # monthly min, monthly max
        (_, _, (salinity_monthly_min, salinity_monthly_max)) = calc_timeseries_params(
            salinity_daily, "m", "salinity"
        )
        print("Generated so data")

        # CURRENT
        current_daily = resampled_xa_das_dict["current"]
        # annual average
        current_annual_average, _, _ = calc_timeseries_params(
            current_daily, "y", "current"
        )
        # monthly min, monthly max
        (_, _, (current_monthly_min, current_monthly_max)) = calc_timeseries_params(
            current_daily, "m", "current"
        )
        print("Generated current data")

        # BATHYMETRY
        bathymetry_climate_res = resampled_xa_das_dict["bathymetry"]
        print("Generated bathymetry data")

        # ERA5
        solar_daily = resampled_xa_das_dict["ssr"]
        # annual average
        solar_annual_average, _, _ = calc_timeseries_params(
            solar_daily, "y", "net_solar"
        )
        # monthly min, monthly max
        _, _, (solar_monthly_min, solar_monthly_max) = calc_timeseries_params(
            solar_daily, "m", "net_solar"
        )
        print("Generated solar data")

        # GT
        gt_climate_res = resampled_xa_das_dict["gt"]
        print("Generated ground truth data")

        merge_list = [
            thetao_annual_average,
            thetao_annual_range,
            thetao_monthly_min,
            thetao_monthly_max,
            thetao_monthly_stdev,
            thetao_weekly_min,
            thetao_weekly_max,
            salinity_annual_average,
            salinity_monthly_min,
            salinity_monthly_max,
            current_annual_average,
            current_monthly_min,
            current_monthly_max,
            solar_annual_average,
            solar_monthly_min,
            solar_monthly_max,
            gt_climate_res,
            bathymetry_climate_res,
        ]
        for xa_da in merge_list:
            if "grid_mapping" in xa_da.attrs:
                del xa_da.attrs["grid_mapping"]
        # MERGE
        merged = xa.merge(merge_list)
        merged.to_netcdf(save_path)
        return merged

    else:
        print(f"{save_path} already exists.")
        return xa.open_dataset(save_path)


def return_time_grouping_offset(period: str):
    """
    Returns the time grouping and offset for a given period.

    Parameters
    ----------
    period (str): The time period for grouping (e.g., 'year', 'month', 'week').

    Returns
    -------
    group (str): The time grouping for the given period.
    offset (str): The offset for the given period.
    """

    if period.lower() in ["year", "y", "annual"]:
        group = "time.year"
        offset = "AS"
    elif period.lower() in ["month", "m"]:
        group = "time.month"
        offset = "MS"
    elif period.lower() in ["week", "w"]:
        group = "time.week"
        offset = "W"

    return group, offset


def calc_weighted_mean(xa_da_daily_means: xa.DataArray, period: str):
    """
    Calculates the weighted mean of daily means for a given period.

    Parameters
    ----------
    xa_da_daily_means (xarray.DataArray): The input xarray DataArray of daily means.
    period (str): The time period for grouping (e.g., 'year', 'month', 'week').

    Returns
    -------
    xarray.DataArray: The weighted mean values for the given period.
    """
    group, offset = return_time_grouping_offset(period)
    # determine original crs
    crs = xa_da_daily_means.rio.crs

    # Determine the month length (has no effect on other time periods)
    month_length = xa_da_daily_means.time.dt.days_in_month
    # Calculate the monthly weights
    weights = month_length.groupby(group) / month_length.groupby(group).sum()

    # Setup our masking for nan values
    ones = xa.where(xa_da_daily_means.isnull(), 0.0, 1.0)

    # Calculate the numerator
    xa_da_daily_means_sum = (
        (xa_da_daily_means * weights).resample(time=offset).sum(dim="time")
    )
    # Calculate the denominator
    ones_out = (ones * weights).resample(time=offset).sum(dim="time")

    # weighted average
    return (xa_da_daily_means_sum / ones_out).rio.write_crs(crs, inplace=True)


def calculate_magnitude(
    horizontal_data: xa.DataArray, vertical_data: xa.DataArray
) -> xa.DataArray:
    """
    Calculates the resultant magnitude of horizontal and vertical data.

    Parameters
    ----------
    horizontal_data (xarray.DataArray): The input xarray DataArray of horizontal data.
    vertical_data (xarray.DataArray): The input xarray DataArray of vertical data.

    Returns
    -------
    xarray.DataArray: The magnitude of the horizontal and vertical data.
    """
    # determine original crs
    crs_h, crs_v = horizontal_data.rio.crs, vertical_data.rio.crs
    if crs_h != crs_v:
        raise ValueError(
            f"Unmatching crs values in xa.DataArrays: horizontal crs = {crs_h}, vertical crs = {crs_v}"
        )

    def magnitude(horizontal_data, vertical_data) -> xa.DataArray:
        return np.sqrt(horizontal_data**2 + vertical_data**2)

    # func = lambda horizontal_data, vertical_data: np.sqrt(
    #     horizontal_data**2 + vertical_data**2
    # )
    # return xa.apply_ufunc(func, horizontal_data, vertical_data)
    return xa.apply_ufunc(magnitude, horizontal_data, vertical_data).rio.write_crs(
        crs_h, inplace=True
    )


def create_train_metadata(
    name: str,
    model_path: Path | str,
    model_type: str,
    data_type: str,
    randomised_search_time: float,
    fit_time: float,
    test_fraction: float,
    features: list[str],
    resolution: float,
) -> None:
    metadata = {
        "model name": name,
        "model path": str(model_path),
        "model type": model_type,
        "data type": data_type,
        "hyperparameter tune time": randomised_search_time,
        "model fit time (s)": fit_time,
        "test fraction": test_fraction,
        "features": features,
        "approximate spatial resolution": resolution,
    }
    out_path = Path(model_path).parent / f"{name}_metadata.json"
    out_file = open(out_path, "w")
    json.dump(metadata, out_file, indent=4)
    print(f"{name} metadata saved to {out_path}")


def initialise_model(model_type: str, random_state: int = 42):
    # continuous models
    data_type = "continuous"
    if model_type == "rf_reg":
        model = RandomForestRegressor(verbose=1, random_state=random_state)
        search_grid = rf_search_grid()
    elif model_type == "brt":
        model = GradientBoostingRegressor(verbose=1, random_state=random_state)
        search_grid = boosted_regression_search_grid()

    # discrete models
    elif model_type == "maxent":
        model = LogisticRegression(verbose=1, random_state=random_state)
        data_type = "discrete"
        search_grid = maximum_entropy_search_grid()
    elif model_type == "rf_cla":
        model = RandomForestClassifier(verbose=1, random_state=random_state)
        data_type = "discrete"
        search_grid = rf_search_grid()

    return model, data_type, search_grid


def train_tune(
    all_data: xa.Dataset,
    model_type: str,
    name: str = "_",
    runs_n: int = 10,
    test_fraction: float = 0.25,
    save_dir: Path | str = None,
    n_iter: int = 10,
    cv: int = 3,
):
    model, data_type, search_grid = initialise_model(model_type)

    (
        X_train,
        X_test,
        y_train,
        y_test,
        train_coordinates,
        test_coordinates,
    ) = spatial_split_train_test(
        all_data,
        "gt",
        data_type="discrete",
        split_type="pixel",
        test_fraction=test_fraction,
    )

    if data_type == "discrete":
        y_train, y_test = model_results.threshold_array(
            y_train
        ), model_results.threshold_array(y_test)

    # register_ray()

    start_time = time.time()
    model_random = RandomizedSearchCV(
        estimator=model,
        param_distributions=search_grid,
        n_iter=n_iter,
        cv=cv,
        verbose=2,
        random_state=42,
        n_jobs=-1,
    )
    end_time = time.time()
    randomised_search_time = end_time - start_time

    # with joblib.parallel_backend("ray"):
    start_time = time.time()
    model_random.fit(X_train, y_train)
    end_time = time.time()
    fit_time = end_time - start_time

    resolution = np.mean(spatial_data.calculate_spatial_resolution(all_data))

    # save best parameters
    if not save_dir:
        save_dir = file_ops.guarantee_existence(
            directories.get_datasets_dir() / "model_params"
        )

    save_path = save_sklearn_model(model_random, save_dir, name)
    create_train_metadata(
        name=name,
        model_path=save_path,
        model_type=model_type,
        data_type=data_type,
        randomised_search_time=randomised_search_time,
        fit_time=fit_time,
        test_fraction=test_fraction,
        features=list(all_data.data_vars),
        resolution=resolution,
    )
    # test
    run_outcomes = n_random_runs_preds(
        model=model_random,
        data_type=data_type,
        runs_n=10,
        xa_ds=all_data,
        test_fraction=test_fraction,
    )

    return run_outcomes


def train_tune_across_resolutions(
    model_type: str,
    d_resolutions: list[float],
    runs_n: int = 10,
    test_fraction: float = 0.25,
):
    resolutions_dict = {}
    for res in tqdm(d_resolutions, desc="Training models at different resolutions"):
        # load in correct-resolution dataset
        res_string = utils.replace_dot_with_dash(f"{res:.05f}d")
        dir = directories.get_comparison_dir() / f"{res_string}_arrays"
        filename = f"all_{res_string}_comparative"
        comparison_file = (dir / filename).with_suffix(".nc")
        all_data = xa.open_dataset(comparison_file)

        run_outcomes = train_tune(
            all_data,
            model_type=model_type,
            name=f"{model_type}_{res_string}",
            runs_n=runs_n,
            test_fraction=test_fraction,
        )

        resolutions_dict[f"{res:.05f}"] = run_outcomes

    return resolutions_dict


def train_test_across_models(model_types: list[str], d_resolution: float = 0.03691):
    model_comp_dir = file_ops.guarantee_existence(
        directories.get_datasets_dir() / "model_params/best_models"
    )

    all_data = get_comparison_xa_ds(d_resolution=d_resolution)
    res_string = utils.replace_dot_with_dash(f"{d_resolution:.05f}d")
    all_model_outcomes = []
    for model in tqdm(
        model_types, total=len(model_types), desc="Fitting each model via random search"
    ):
        run_outcomes = train_tune(
            all_data=all_data,
            model_type=model,
            save_dir=model_comp_dir,
            name=f"{model}_{res_string}_tuned",
            runs_n=10,
            test_fraction=0.25,
        )
        all_model_outcomes.append(run_outcomes)

    return all_model_outcomes


def get_comparison_xa_ds(d_resolution: float = 0.03691):
    res_string = utils.replace_dot_with_dash(f"{d_resolution:.05f}d")
    all_data_dir = directories.get_comparison_dir() / f"{res_string}_arrays"
    all_data_name = f"all_{res_string}_comparative"
    return xa.open_dataset((all_data_dir / all_data_name).with_suffix(".nc"))
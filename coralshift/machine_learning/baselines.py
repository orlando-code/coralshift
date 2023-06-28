from __future__ import annotations

import pickle
import json
import time
import xarray as xa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# from matplotlib.axes import Axes
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
from coralshift.dataloading import bathymetry


def generate_test_train_coordinates(
    xa_ds: xa.Dataset,
    split_type: str = "pixel",
    test_lats: tuple[float] = None,
    test_lons: tuple[float] = None,
    test_fraction: float = 0.2,
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

    if split_type == "pixel":
        # have to handle time: make sure sampling spatially rather than spatiotempoorally
        train_coordinates, test_coordinates = spatial_data.generate_coordinate_pairs(
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

            train_coordinates, _ = spatial_data.generate_coordinate_pairs(train_xa, 0)
            test_coordinates, _ = spatial_data.generate_coordinate_pairs(test_xa, 0)

        # if specific latitude/longitude boundary specified, cut out test region and train on all else
        else:
            test_xa = xa_ds.isel(
                {
                    "latitude": slice(test_lats[0], test_lats[1]),
                    "longitude": slice(test_lons[0], test_lons[1]),
                }
            )
            all_coordinates = spatial_data.generate_coordinate_pairs(xa_ds)

            test_coordinates = spatial_data.generate_coordinate_pairs(test_xa)
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


# def xa_dss_to_df(
#     xa_dss: list[xa.Dataset],
#     split_type: str = "pixel",
#     test_lats: tuple[float] = None,
#     test_lons: tuple[float] = None,
#     test_fraction: float = 0.2,
#     bath_mask: bool = True,
#     ignore_vars: list = ["spatial_ref", "band", "depth"],
# ):
#     train_coords, test_coords, dfs = [], [], []
#     for xa_ds in xa_dss:
#         # compute out dasked chunks, fill Nan values with 0, drop columns which would confuse model
#         df = (
#             xa_ds.stack(points=("latitude", "longitude", "time"))
#             .compute()
#             .astype("float32")
#             .to_dataframe()
#         )
#         df["onehotnan"] = df.isnull().any(axis=1).astype(int)
#         # fill nans with 0 and drop datetime columns
#         df = df.fillna(0).drop(
#             columns=list(df.select_dtypes(include="datetime64").columns)
#         )
#         # drop ignored vars
#         df = df.drop(columns=list(set(ignore_vars).intersection(df.columns)))

#         train_coordinates, test_coordinates = generate_test_train_coordinates(
#             xa_ds, split_type, test_lats, test_lons, test_fraction, bath_mask
#         )

#         train_coords.extend(train_coordinates)
#         test_coords.extend(test_coordinates)
#         dfs.append(df)

#     # flatten dataset for row indexing and model training
#     return pd.concat(dfs), train_coords, test_coords


def generate_test_train_coords_from_df(
    df: pd.DataFrame,
    test_fraction: float = 0.25,
    split_type: str = "pixel",
    train_test_lat_divide: int = float,
    train_direction: str = "N",
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Shuffle the filtered DataFrame randomly
    df_shuffled = df.sample(frac=1, random_state=random_seed)

    if split_type == "pixel":
        # num datapoints
        num_samples = len(df)
        # Calculate the split sizes
        test_size = int(num_samples * test_fraction)
        train_size = num_samples - test_size
        # Split the coordinates into two lists based on the split sizes
        train_coordinates = (
            df_shuffled[["latitude", "longitude"]].values[:train_size].tolist()
        )
        test_coordinates = (
            df_shuffled[["latitude", "longitude"]]
            .values[train_size : train_size + test_size]  # noqa
            .tolist()
        )
    elif split_type == "spatial":
        if train_direction == "S":
            train_rows = df_shuffled.loc[
                (
                    train_test_lat_divide
                    >= df_shuffled.index.get_level_values("latitude")
                )
            ]
            test_rows = df_shuffled.loc[
                (
                    train_test_lat_divide
                    <= df_shuffled.index.get_level_values("latitude")
                )
            ]
        elif train_direction == "N":
            train_rows = df_shuffled.loc[
                (
                    train_test_lat_divide
                    <= df_shuffled.index.get_level_values("latitude")
                )
            ]
            test_rows = df_shuffled.loc[
                (
                    train_test_lat_divide
                    >= df_shuffled.index.get_level_values("latitude")
                )
            ]
        else:
            raise ValueError(f"Train direction: {train_direction} not recognised.")

        train_coordinates = train_rows[["latitude", "longitude"]].values.tolist()
        test_coordinates = test_rows[["latitude", "longitude"]].values.tolist()
    else:
        print(f"Unrecognised split type {split_type}.")

    return train_coordinates, test_coordinates


def vary_train_test_lat(
    df: pd.DataFrame,
    num_vals: int = 10,
    train_direction: str = "N",
    random_seed: int = 42,
) -> list[tuple[float, float]]:
    lat_min, lat_max = df.index["latitude"].min(), df.index["latitude"].max()
    lat_dividers = np.linspace(lat_min, lat_max, num_vals)

    train_coords_list, test_coords_list = [], []
    for lat_divide in lat_dividers:
        train_coords, test_coords = generate_test_train_coords_from_df(
            df=df,
            split_type="pixel",
            train_test_lat_divide=lat_divide,
            train_direction=train_direction,
            random_seed=random_seed,
        )
        train_coords_list.append(train_coords)
        test_coords_list.append(test_coords)
    return lat_divide, train_coords_list, test_coords_list


def generate_test_train_coords_from_dfs(
    dfs: list[pd.DataFrame],
    test_fraction: float = 0.25,
    split_type: str = "pixel",
    train_test_lat_divide: int = float,
    train_direction: str = "N",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_coords_list, test_coords_list = [], []
    for df in dfs:
        lists = generate_test_train_coords_from_df(
            df, test_fraction, split_type, train_test_lat_divide, train_direction
        )
        train_coords_list.append(lists[0])
        test_coords_list.append(lists[1])

    return train_coords_list, test_coords_list


def process_df_for_ml(
    df: pd.DataFrame, ignore_vars: list[str], drop_all_nans: bool = True
) -> pd.DataFrame:
    # drop ignored vars
    df = df.drop(columns=list(set(ignore_vars).intersection(df.columns)))

    if drop_all_nans:
        # remove rows which are all nans
        df = utils.drop_nan_rows(df)
    # onehot encoode any remaining nans
    df["onehotnan"] = df.isnull().any(axis=1).astype(int)
    # fill nans with 0
    df = df.fillna(0)

    # flatten dataset for row indexing and model training
    return df


def xa_dss_to_df(
    xa_dss: list[xa.Dataset],
    bath_mask: bool = True,
    ignore_vars: list = ["spatial_ref", "band", "depth"],
    drop_all_nans: bool = True,
):
    dfs = []
    for xa_ds in xa_dss:
        if bath_mask:
            # set all values outside of the shallow water region to nan for future omission
            shallow_mask = spatial_data.generate_var_mask(xa_ds)
            xa_ds = xa_ds.where(shallow_mask, np.nan)

        # compute out dasked chunks, send type to float32, stack into df, drop any datetime columns
        df = (
            xa_ds.stack(points=("latitude", "longitude"))
            .compute()
            .astype("float32")
            .to_dataframe()
        )
        # drop temporal columns
        df = df.drop(columns=list(df.select_dtypes(include="datetime64").columns))
        df = process_df_for_ml(df, ignore_vars=ignore_vars, drop_all_nans=drop_all_nans)

        # # generate train_test_coordinates
        # train_coordinates, test_coordinates = generate_test_train_coordinates(
        #     xa_ds, split_type, test_lats, test_lons, test_fraction, bath_mask
        # )

        # train_coords.append(train_coordinates)
        # test_coords.append(test_coordinates)
        dfs.append(df)
    return dfs


def spatial_split_train_test(
    xa_dss: list[xa.Dataset],
    gt_label: str = "gt",
    data_type: str = "continuous",
    ignore_vars: list = ["spatial_ref", "band", "depth"],
    split_type: str = "pixel",
    test_fraction: float = 0.25,
    train_test_lat_divide: int = float,
    train_direction: str = "N",
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
    # send input to list if not already
    xa_dss = utils.cast_to_list(xa_dss)
    # flatten datasets to pandas dataframes and process
    flattened_data_dfs = xa_dss_to_df(xa_dss, bath_mask=bath_mask)
    # generate training and testing coordinates
    train_coords_list, test_coords_list = generate_test_train_coords_from_dfs(
        flattened_data_dfs, test_fraction=test_fraction
    )

    # normalise dataframe via min/max scaling
    normalised_dfs = [
        (flattened_data - flattened_data.min())
        / (flattened_data.max() - flattened_data.min())
        for flattened_data in flattened_data_dfs
    ]

    train_rows, test_rows = [], []
    X_trains, X_tests, y_trains, y_tests = [], [], [], []
    for i in range(len(flattened_data_dfs)):
        # return train and test data rows from dataframe
        train_rows = utils.select_df_rows_by_coords(
            normalised_dfs[i], train_coords_list[i]
        )
        test_rows = utils.select_df_rows_by_coords(
            normalised_dfs[i], test_coords_list[i]
        )
        # determine the corresponding labels
        y_train, y_test = train_rows["gt"], test_rows["gt"]
        if data_type == "discrete":
            y_train, y_test = threshold_array(y_train), threshold_array(y_test)
        # append everything to where it needs to be
        X_trains.append(train_rows.drop("gt", axis=1))
        X_tests.append(test_rows.drop("gt", axis=1))
        y_trains.append(y_train), y_tests.append(y_test)

    # for now, merge all lists together
    X_trains = utils.flatten_list(X_trains)
    X_tests = utils.flatten_list(X_tests)
    y_trains = utils.flatten_list(y_trains)
    y_tests = utils.flatten_list(y_tests)
    train_coords_list = utils.flatten_list(train_coords_list)
    test_coords_list = utils.flatten_list(test_coords_list)

    return X_trains, X_tests, y_trains, y_tests, train_coords_list, test_coords_list


# def spatial_split_train_test(
#     xa_dss: list[xa.Dataset],
#     gt_label: str = "gt",
#     data_type: str = "continuous",
#     ignore_vars: list = ["spatial_ref", "band", "depth"],
#     split_type: str = "pixel",
#     test_lats: tuple[float] = None,
#     test_lons: tuple[float] = None,
#     test_fraction: float = 0.25,
#     bath_mask: bool = True,
# ) -> tuple:
#     """
#     Split the input dataset into train and test sets based on spatial coordinates.

#     Parameters
#     ----------
#         xa_ds (xa.Dataset): The input xarray Dataset.
#         gt_label: The ground truth label.
#         ignore_vars (list): A list of variables to ignore during splitting. Default is
#             ["time", "spatial_ref", "band", "depth"].
#         split_type (str): The split type, either "pixel" or "region". Default is "pixel".
#         test_lats (tuple[float]): The latitude range for the test region. Required for "region" split type.
#             Default is None.
#         test_lons (tuple[float]): The longitude range for the test region. Required for "region" split type.
#             Default is None.
#         test_fraction (float): The fraction of data to be used for the test set. Default is 0.2.

#     Returns
#     -------
#         tuple: A tuple containing X_train, X_test, y_train, and y_test.
#     """
#     # send input to list if not already
#     xa_dss = utils.list_if_not_already(xa_dss)
#     flattened_data, train_coords, test_coords = xa_dss_to_df(
#         xa_dss,
#         split_type=split_type,
#         test_lats=test_lats,
#         test_lons=test_lons,
#         test_fraction=test_fraction,
#         bath_mask=bath_mask,
#     )
#     # normalise data via min/max scaling
#     normalised_data = (flattened_data - flattened_data.min()) / (
#         flattened_data.max() - flattened_data.min()
#     )

#     # return train and test rows from dataframe
#     train_rows = utils.select_df_rows_by_coords(normalised_data, train_coords)
#     test_rows = utils.select_df_rows_by_coords(normalised_data, test_coords)

#     # assign rows to test and train features/labels
#     X_train, X_test = train_rows.drop("gt", axis=1), test_rows.drop("gt", axis=1)
#     y_train, y_test = train_rows["gt"], test_rows["gt"]

#     if data_type == "discrete":
#         y_train, y_test = threshold_array(y_train), threshold_array(y_test)

#     return X_train, X_test, y_train, y_test, train_coords, test_coords


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
        binary_y_labels, binary_predictions = threshold_label(
            outcome[0], outcome[1], binarize_threshold
        )

        fpr, tpr, _ = sklmetrics.curve(
            binary_y_labels, binary_predictions, drop_intermediate=False
        )
        roc_auc = sklmetrics.auc(fpr, tpr)

        label = f"{roc_auc:.02f}"
        ax.plot(fpr, tpr, label=label, color=colors[c])

    n_runs = len(run_outcomes)
    # format
    model_results.format_roc(
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
    if save_path.is_file():
        save_path = (Path(savedir) / f"{filename}_new").with_suffix(".pickle")
        print(f"Model at {save_path} already exists.")
    # save model
    with open(save_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved model to {save_path}.")
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
    model_results.format_roc(
        ax=ax,
        title="Receiver Operating Characteristic (ROC) Curve\nfor several coral presence/absence thresholds",
    )
    ax.legend(title="threshold value | auc")


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
    result = np.where(np.array(array) > threshold, 1, 0)
    if isinstance(array, pd.Series):
        return pd.Series(result, index=array.index)
    else:
        return pd.Series(result)


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
    min_val = (
        xa_da_daily_means.min(dim="time", skipna=True)
        .rename(base_name + "min")
        .rio.write_crs(crs, inplace=True)
    )
    max_val = (
        xa_da_daily_means.max(dim="time", skipna=True)
        .rename(base_name + "max")
        .rio.write_crs(crs, inplace=True)
    )

    # Return the weighted average
    return weighted_av, stdev, (min_val, max_val)


def generate_reproducing_metrics(
    resampled_xa_das_dict: dict, target_resolution_d: float = None, region: str = None
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
    if target_resolution_d:
        resolution = target_resolution_d
    else:
        resolution = np.mean(
            spatial_data.calculate_spatial_resolution(resampled_xa_das_dict["thetao"])
        )

    res_string = utils.replace_dot_with_dash(f"{resolution:.04f}d")

    if not region:
        filename = f"{res_string}_arrays/all_{res_string}_comparative"
    else:
        region_dir = file_ops.guarantee_existence(
            directories.get_comparison_dir()
            / f"{bathymetry.ReefAreas().get_short_filename(region)}/{res_string}_arrays"
        )
        # region_letter = bathymetry.ReefAreas().get_letter(region)
        filename = region_dir / f"all_{res_string}_comparative"

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
        print("Generated thetao data")

        # SALINITY
        salinity_daily = resampled_xa_das_dict["so"]
        # annual average
        salinity_annual_average, _, _ = calc_timeseries_params(
            salinity_daily, "y", "salinity"
        )
        # monthly min, monthly max
        (
            _,
            _,
            (salinity_monthly_min, salinity_monthly_max),
        ) = calc_timeseries_params(salinity_daily, "m", "salinity")
        print("Generated so data")

        # CURRENT
        current_daily = resampled_xa_das_dict["current"]
        # annual average
        current_annual_average, _, _ = calc_timeseries_params(
            current_daily, "y", "current"
        )
        # monthly min, monthly max
        (
            _,
            _,
            (current_monthly_min, current_monthly_max),
        ) = calc_timeseries_params(current_daily, "m", "current")
        print("Generated current data")

        # BATHYMETRY
        bathymetry_climate_res = resampled_xa_das_dict["bathymetry"]
        print("Generated bathymetry data")

        # # ERA5
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
            thetao_annual_average.mean(dim="time"),
            thetao_annual_range,
            thetao_monthly_min,
            thetao_monthly_max,
            thetao_monthly_stdev,
            thetao_weekly_min,
            thetao_weekly_max,
            salinity_annual_average.mean(dim="time"),
            salinity_monthly_min,
            salinity_monthly_max,
            current_annual_average.mean(dim="time"),
            current_monthly_min,
            current_monthly_max,
            solar_annual_average.mean(dim="time"),
            solar_monthly_min,
            solar_monthly_max,
            gt_climate_res,
            bathymetry_climate_res,
        ]
        for xa_da in merge_list:
            if "grid_mapping" in xa_da.attrs:
                del xa_da.attrs["grid_mapping"]
        # MERGE
        merged = xa.merge(merge_list).astype(np.float64)
        merged.attrs["region"] = region
        with np.errstate(divide="ignore", invalid="ignore"):
            merged.to_netcdf(save_path)
            return merged

    else:
        print(f"{save_path} already exists.")
        return xa.open_dataset(save_path, decode_coords="all")


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
        offset = pd.offsets.Week()

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
    n_iter: int,
    cv: int,
    param_distributions: dict,
    random_state: int,
    n_jobs: int,
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
        "number of fit iterations": n_iter,
        "cross validation fold size": cv,
        "random_state": 42,
        "n_jobs": -1,
    }
    # update metadata with parameter search grid
    metadata.update(param_distributions)

    filename = f"{name}_metadata"
    save_path = (Path(model_path).parent / filename).with_suffix(".json")
    if save_path.is_file():
        save_path = (Path(model_path).parent / f"{filename}_new").with_suffix(".json")
        print(f"Metadata at {save_path} already exists.")
    out_file = open(save_path, "w")
    json.dump(metadata, out_file, indent=4)
    print(f"{name} metadata saved to {save_path}")


class ModelInitializer:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        # self.data_type = None
        # self.model = None
        # self.search_grid = None

        self.model_info = [
            # continuous model
            {
                "model_type": "rf_reg",
                "data_type": "continuous",
                "model": RandomForestRegressor(
                    verbose=1, random_state=self.random_state
                ),
                "search_grid": rf_search_grid(),
            },
            {
                "model_type": "brt",
                "data_type": "continuous",
                "model": GradientBoostingRegressor(
                    verbose=1, random_state=self.random_state
                ),
                "search_grid": boosted_regression_search_grid(),
            },
            # discrete models
            {
                "model_type": "rf_cla",
                "data_type": "discrete",
                "model": RandomForestClassifier(
                    class_weight="balanced", verbose=1, random_state=self.random_state
                ),
                "search_grid": rf_search_grid(),
            },
            {
                "model_type": "maxent",
                "data_type": "discrete",
                "model": LogisticRegression(
                    class_weight="balanced", verbose=1, random_state=self.random_state
                ),
                "search_grid": maximum_entropy_search_grid(),
            },
        ]

    def get_data_type(self, model_type):
        for m in self.model_info:
            if m["model_type"] == model_type:
                return m["data_type"]
        else:
            raise ValueError(f"'{model_type}' not a valid model.")

    def get_model(self, model_type):
        for m in self.model_info:
            if m["model_type"] == model_type:
                return m["model"]
        else:
            raise ValueError(f"'{model_type}' not a valid model.")

    def get_search_grid(self, model_type):
        for m in self.model_info:
            if m["model_type"] == model_type:
                return m["search_grid"]
        else:
            raise ValueError(f"'{model_type}' not a valid model.")


def initialise_model(model_type: str, random_state: int = 42):
    model_instance = ModelInitializer(random_state=random_state)

    return (
        model_instance.get_model(model_type),
        model_instance.get_data_type(model_type),
        model_instance.get_search_grid(model_type),
    )


def calculate_class_weight(label_array: np.ndarray):
    unique_values, counts = np.unique(label_array, return_counts=True)
    occurrence_dict = dict(zip(unique_values, counts))
    return occurrence_dict


def train_tune(
    X_train,
    y_train,
    model_type: str,
    resolution: float,
    name: str = "_",
    test_fraction: float = 0.25,
    save_dir: Path | str = None,
    n_iter: int = 50,
    cv: int = 3,
):
    model, data_type, search_grid = initialise_model(model_type)

    if data_type == "discrete":
        y_train = threshold_array(y_train)
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

    print("Fitting model with a randomized hyperparameter search...")
    # with joblib.parallel_backend("ray"):
    start_time = time.time()
    model_random.fit(X_train, y_train)
    end_time = time.time()
    fit_time = end_time - start_time

    # save best parameter model and metadata
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
        features=list(X_train.columns),
        resolution=resolution,
        n_iter=n_iter,
        cv=cv,
        param_distributions=search_grid,
        random_state=42,
        n_jobs=-1,
    )


def train_tune_across_resolutions(
    model_type: str,
    d_resolutions: list[float],
    split_type: str = "pixel",
    test_lats: tuple[float] = None,
    test_lons: tuple[float] = None,
    runs_n: int = 10,
    test_fraction: float = 0.25,
):
    # TODO: finish this
    for d_res in tqdm(
        d_resolutions,
        total=len(d_resolutions),
        desc="Training models at different resolutions",
    ):
        # load in correct-resolution dataset
        res_string = utils.replace_dot_with_dash(f"{d_res:.04f}d")

        dir = directories.get_comparison_dir() / f"{res_string}_arrays"
        filename = f"all_{res_string}_comparative"

        all_data = xa.open_dataset(comparison_file_path, decode_coords="all")

        # define train/test split so it's the same for all models
        (X_train, X_test, y_train, y_test, _, _) = spatial_split_train_test(
            all_data,
            "gt",
            split_type=split_type,
            test_fraction=test_fraction,
        )

        comparison_file_path = (dir / filename).with_suffix(".nc")

        train_tune(
            X_train,
            y_train,
            model_type=model_type,
            resolution=d_res,
            save_dir=model_comp_dir,
            name=f"{model_type}_{res_string}_tuned",
            test_fraction=0.25,
        )


def train_tune_across_models(
    model_types: list[str],
    d_resolution: float = 0.03691,
    split_type: str = "pixel",
    test_lats: tuple[float] = None,
    test_lons: tuple[float] = None,
    test_fraction: float = 0.25,
    cv: int = 3,
    n_iter: int = 10,
):
    model_comp_dir = file_ops.guarantee_existence(
        directories.get_datasets_dir() / "model_params/best_models"
    )

    all_data = get_comparison_xa_ds(d_resolution=d_resolution)
    res_string = utils.replace_dot_with_dash(f"{d_resolution:.04f}d")

    # define train/test split so it's the same for all models
    (X_trains, X_tests, y_trains, y_tests, _, _) = spatial_split_train_test(
        all_data,
        "gt",
        split_type=split_type,
        test_fraction=test_fraction,
    )

    for model in tqdm(
        model_types, total=len(model_types), desc="Fitting each model via random search"
    ):
        train_tune(
            X_train,
            y_train,
            model_type=model,
            resolution=d_resolution,
            save_dir=model_comp_dir,
            name=f"{model}_{res_string}_tuned",
            test_fraction=test_fraction,
            cv=cv,
            n_iter=n_iter,
        )


def get_comparison_xa_ds(
    region_list: list = ["A", "B", "C", "D"], d_resolution: float = 0.0368
):
    res_string = utils.replace_dot_with_dash(f"{d_resolution:.04f}d")

    xa_dss = []
    for region in region_list:
        region_name = bathymetry.ReefAreas().get_short_filename(region)
        all_data_dir = (
            directories.get_comparison_dir() / f"{region_name}/{res_string}_arrays"
        )
        all_data_name = f"all_{res_string}_comparative"
        xa_dss.append(
            xa.open_dataset(
                (all_data_dir / all_data_name).with_suffix(".nc"), decode_coords="all"
            )
        )

    return xa_dss


def generate_reproducing_metrics_for_regions(
    regions_list: list = ["A", "B", "C", "D"], target_resolution_d: float = 1 / 27
) -> None:
    for region in tqdm(
        regions_list,
        total=len(regions_list),
        position=0,
        leave=False,
        desc=" Processing regions",
    ):
        lat_lims = bathymetry.ReefAreas().get_lat_lon_limits(region)[0]
        lon_lims = bathymetry.ReefAreas().get_lat_lon_limits(region)[1]

        # create list of xarray dataarrays
        reproduction_xa_list = load_and_process_reproducing_xa_das(region)
        # create dictionary of xa arrays, resampled to correct resolution
        resampled_xa_das_dict = file_ops.resample_list_xa_ds_into_dict(
            reproduction_xa_list,
            target_resolution=target_resolution_d,
            unit="d",
            lat_lims=lat_lims,
            lon_lims=lon_lims,
        )
        # generate and save reproducing metrics from merged dict
        generate_reproducing_metrics(
            resampled_xa_das_dict,
            region=region,
            target_resolution_d=target_resolution_d,
        )


def load_and_process_reproducing_xa_das(
    region: str, chunk_dict: dict = {"latitude": 100, "longitude": 100, "time": 100}
) -> list[xa.DataArray]:
    """
    Load and process xarray data arrays for reproducing metrics.

    Returns
    -------
        list[xa.DataArray]: A list containing the processed xarray data arrays.
    """
    region_name = bathymetry.ReefAreas().get_short_filename(region)
    region_letter = bathymetry.ReefAreas().get_letter(region)

    # load in daily sea water potential temp
    # thetao_daily = xa.open_dataarray(directories.get_processed_dir() / "arrays/thetao.nc")

    dailies_array = xa.open_dataset(
        directories.get_daily_cmems_dir()
        / f"{region_name}/cmems_gopr_daily_{region_letter}.nc",
        decode_coords="all",
        chunks=chunk_dict,
    ).isel(depth=0)

    # load in daily sea water potential temp
    thetao_daily = dailies_array["thetao"]
    # load in daily sea water salinity means
    salinity_daily = dailies_array["so"]
    # calculate current magnitude
    current_daily = calculate_magnitude(
        dailies_array["uo"].compute(), dailies_array["vo"].compute()
    ).rename("current")
    # TODO: download ERA5 files
    # load bathymetry file TODO: separate function for this
    bath_file = list(
        directories.get_bathymetry_datasets_dir().glob(f"{region_name}_*.nc")
    )[0]
    bath = xa.open_dataarray(bath_file, decode_coords="all", chunks=chunk_dict).rename(
        "bathymetry"
    )
    # Load in ERA5 surface net solar radiation
    net_solar_file = list(
        (directories.get_era5_data_dir() / f"{region_name}/weather_parameters/").glob(
            "*surface_net_solar_radiation_*.nc"
        )
    )[0]
    net_solar = spatial_data.process_xa_d(
        xa.open_dataarray(net_solar_file, decode_coords="all", chunks=chunk_dict)
    )
    net_solar = net_solar.resample(time="1D").mean(dim="time")

    # Load in ground truth coral data
    gt = xa.open_dataarray(
        directories.get_gt_files_dir() / f"coral_region_{region_letter}_1000m.nc",
        decode_coords="all",
        chunks=chunk_dict,
    ).rename("gt")

    return [thetao_daily, salinity_daily, current_daily, net_solar, bath, gt]

    # load in daily sea water salinity means
    # salinity_daily = xa.open_dataarray(directories.get_processed_dir() / "arrays/so.nc")

    # load in daily latitudinal and longitudinal currents
    # uo_daily = xa.open_dataarray(directories.get_processed_dir() / "arrays/uo.nc")
    # vo_daily = xa.open_dataarray(directories.get_processed_dir() / "arrays/vo.nc")
    # calculate current magnitude
    # current_daily = calculate_magnitude(uo_daily, vo_daily).rename("current")

    # bathymetry = xa.open_dataset(
    #     directories.get_bathymetry_datasets_dir() / "bathymetry_A_0-00030d.nc"
    # ).rio.write_crs("EPSG:4326")["bathymetry_A"]

    # fetch resolution
    # correct bathymetry file
    # bathymetry = xa.open_dataset(directories.get_bathymetry_datasets_dir() / "bathymetry_A_0-00030d.nc")

    # Load in ERA5 surface net solar radiation and upscale to climate variable resolution
    # solar_radiation = xa.open_dataarray(
    #     directories.get_era5_data_dir() / "weather_parameters/VAR_surface_net_solar_radiation_LATS_-10_-17_LONS_142_147_YEAR_1993-2020.nc" # noqa
    #     ).rio.write_crs("EPSG:4326")
    # solar_radiation = xa.open_dataarray(
    #     directories.get_era5_data_dir() / "weather_parameters/VAR_surface_net_solar_radiation_LATS_-10_-17_LONS_142_147_YEAR_1993-2020.nc" # noqa
    #     )
    # average solar_radiation daily
    # solar_radiation_daily = solar_radiation.resample(time="1D").mean(dim="time")


def load_model(
    model_name: Path | str, model_dir: Path | str = directories.get_best_model_dir()
):
    if Path(model_name).is_file():
        model = pickle.load(open(model_name, "rb"))
    else:
        model = pickle.load(open(Path(model_dir) / model_name, "rb"))
    print(model.best_params_)
    return model

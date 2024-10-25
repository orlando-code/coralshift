# spatial
# import geopandas as gpd
import cartopy.crs as ccrs

# import rasterio

# from rasterio import features as featuresio
import xarray as xa

from pyinterp.backends import xarray

# from pyinterp import fill
# import xesmf as xe

# general
import matplotlib.pyplot as plt
import numpy as np

# from datetime import datetime
import pandas as pd
from tqdm.auto import tqdm

# from functools import lru_cache

# file handling
from pathlib import Path
import sys

# import re
# import warnings

# ml
# from sklearn.metrics import mean_squared_error
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split

# from sklearn.ensemble import (
#     RandomForestRegressor,
#     RandomForestClassifier,
#     GradientBoostingRegressor,
#     GradientBoostingClassifier,
# )
# from sklearn.linear_model import LogisticRegression
# from sklearn.neural_network import MLPClassifier, MLPRegressor

# from coralshift.machine_learning import baselines
# from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# import xgboost as xgb

# custom
from coralshift.processing import spatial_data
from coralshift.plotting import spatial_plots

# from coralshift.machine_learning import baselines

# from xgb_funcs.py
###############################################################################


# too specific, but may want to adapt to do something with custom cv implementation
# e.g. https://daniel-furman.github.io/Python-species-distribution-modeling/
# def xgb_random_search(dtrain: xgb.DMatrix, custom_scale_pos_weight=None):

#     param_space = functions_creche.xgb_random_search(custom_scale_pos_weight=None)

#     # time execution
#     start_time = time.time()

#     # for each combination of hyperparameters ()
#     cv_results = xgb.cv(
#         params,
#         dtrain,
#         num_boost_round=num_boost_round,
#         seed=42,
#         nfold=CV_FOLDS,
#         metrics="rmse",
#         early_stopping_rounds=EARLY_STOPPING_ROUNDS,
#     )

#     reg.fit(X, y)
#     end_time = time.time()

#     # Calculate and print the elapsed time
#     elapsed_time = end_time - start_time
#     print(f"Elapsed Time: {elapsed_time} seconds")

#     print(reg.best_score_)
#     print(reg.best_params_)


# from baselines.py
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


def generate_test_train_coords_from_df(
    df: pd.DataFrame,
    test_fraction: float = 0.25,
    split_type: str = "pixel",
    train_test_lat_divide: int = -18,
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


# can definitely do better than this now, but may give a starting point
def spatial_split_train_test(
    xa_dss: list[xa.Dataset],
    gt_label: str = "gt",
    data_type: str = "continuous",
    ignore_vars: list = ["spatial_ref", "band", "depth"],
    split_type: str = "pixel",
    test_fraction: float = 0.25,
    train_test_lat_divide: int = -18,
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
    # xa_dss = utils.cast_to_list(xa_dss)
    # flatten datasets to pandas dataframes and process
    # flattened_data_dfs = xa_dss_to_df(xa_dss, bath_mask=bath_mask)
    # generate training and testing coordinates
    # train_coords_list, test_coords_list = generate_test_train_coords_from_dfs(
    #     flattened_data_dfs,
    #     test_fraction=test_fraction,
    #     split_type=split_type,
    #     train_test_lat_divide=train_test_lat_divide,
    #     train_direction=train_direction,
    # )

    # # normalise dataframe via min/max scaling
    # normalised_dfs = [
    #     (flattened_data - flattened_data.min())
    #     / (flattened_data.max() - flattened_data.min())
    #     for flattened_data in flattened_data_dfs
    # ]

    # train_rows, test_rows = [], []
    # X_trains, X_tests, y_trains, y_tests = [], [], [], []
    # for i in range(len(flattened_data_dfs)):
    #     # return train and test data rows from dataframe
    #     train_rows = utils.select_df_rows_by_coords(
    #         normalised_dfs[i], train_coords_list[i]
    #     )
    #     test_rows = utils.select_df_rows_by_coords(
    #         normalised_dfs[i], test_coords_list[i]
    #     )
    #     # determine the corresponding labels
    #     y_train, y_test = train_rows["gt"], test_rows["gt"]
    #     if data_type == "discrete":
    #         y_train, y_test = threshold_array(y_train), threshold_array(y_test)
    #     # append everything to where it needs to be
    #     X_trains.append(train_rows.drop("gt", axis=1))
    #     X_tests.append(test_rows.drop("gt", axis=1))
    #     y_trains.append(y_train), y_tests.append(y_test)

    # # for now, merge all lists together
    # X_trains = utils.flatten_list(X_trains)
    # X_tests = utils.flatten_list(X_tests)
    # y_trains = utils.flatten_list(y_trains)
    # y_tests = utils.flatten_list(y_tests)
    # train_coords_list = utils.flatten_list(train_coords_list)
    # test_coords_list = utils.flatten_list(test_coords_list)

    # return X_trains, X_tests, y_trains, y_tests, train_coords_list, test_coords_list


def scaling_params_from_df(df: pd.DataFrame, var_cols: list[str]) -> dict:
    """
    Calculate scaling parameters (min, max) for the specified variable columns in a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        var_cols (list[str]): List of column names to calculate scaling parameters for.

    Returns:
        dict: Dictionary with column names as keys and (min, max) tuples as values.

    TODO: expand to other scaling methods
    TODO: useful to have values as list rather than dict?
    """
    scaling_params = df[var_cols].agg(["min", "max"], skipna=True)
    # cast from dataframe to dictionary. Keys: column names, values: (min, max) tuples
    scaling_params_dict = scaling_params.to_dict()

    return scaling_params_dict


# scale dataframe using scaling_params
def scale_dataframe(df: pd.DataFrame, scaling_params: dict) -> pd.DataFrame:
    """
    Scale the specified columns of a DataFrame using the provided scaling parameters.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        scaling_params (dict): Dictionary with column names as keys and (min, max) scaling parameters as values.

    Returns:
        pd.DataFrame: The scaled DataFrame.
    """
    # Create a copy of the original DataFrame to avoid modifying it in place
    scaled_df = df.copy()

    for col, params in scaling_params.items():
        min_val = params.get("min", 0.0)
        max_val = params.get("max", 1.0)
        scaled_df[col] = (scaled_df[col] - min_val) / (max_val - min_val)

    return scaled_df


def investigate_depth_mask(comp_var_xa, mask_var_xa, var_limits: list[tuple[float]]):
    # TODO: make less janky
    raw_vals = []
    positive_negative_ratio = []
    # x_labels = [str(lim_pair) for lim_pair in var_limits]

    for lim_pair in var_limits:
        masked_vals = generate_var_mask(comp_var_xa, mask_var_xa, lim_pair)
        val_sum = np.nansum(masked_vals["elevation"].values)
        raw_vals.append(val_sum)

        positive_negative_ratio.append(
            val_sum / np.count_nonzero(masked_vals["elevation"].values == 0)
        )

        utils.calc_non_zero_ratio(masked_vals["elevation_values"])
    return raw_vals, positive_negative_ratio


def generate_var_mask(
    comp_var_xa: xa.DataArray,
    mask_var_xa: xa.DataArray,
    limits: tuple[float] = [-2000, 0],
) -> xa.DataArray:
    mask = (mask_var_xa <= max(limits)) & (mask_var_xa >= min(limits))
    return comp_var_xa.where(mask, drop=True)


def split_dataset_and_save(
    ds_fp, divisor, output_dir_name: str = None, select_vars: list[str] = None
):
    ds = xa.open_dataset(ds_fp)

    if select_vars:
        ds = ds[select_vars]

    subsets_dict = split_dataset_by_indices(ds, divisor)

    # Create a subdirectory to save the split datasets
    if output_dir_name:
        output_dir = (
            Path(ds_fp).parent / f"{output_dir_name}_{divisor**2}_split_datasets"
        )
    else:
        output_dir = Path(ds_fp).parent / f"{divisor**2}_split_datasets"
    output_dir.mkdir(parents=True, exist_ok=True)

    for coord_info, subset in tqdm(
        subsets_dict.items(), desc="saving dataset subsets..."
    ):
        stem_stem = str(Path(ds_fp).stem).split("lats")[0]
        # Construct the filename based on bounds
        filename = f"{stem_stem}_{coord_info}.nc"
        save_fp = output_dir / filename
        subset.to_netcdf(save_fp)
    return subsets_dict


def split_dataset_by_indices(dataset, divisor) -> dict:
    subsets_dict = {}
    num_lats = len(dataset.latitude.values) // divisor
    num_lons = len(dataset.longitude.values) // divisor
    for i in range(divisor):
        for j in range(divisor):
            start_lat_ind = i * num_lats
            start_lon_ind = j * num_lons

            subset = dataset.isel(
                latitude=slice(start_lat_ind, start_lat_ind + num_lats),
                longitude=slice(start_lon_ind, start_lon_ind + num_lons),
            )

            lat_lims = spatial_data.min_max_of_coords(subset, "latitude")
            lon_lims = spatial_data.min_max_of_coords(subset, "longitude")

            coord_info = lat_lon_string_from_tuples(lat_lims, lon_lims)
            subsets_dict[coord_info] = subset

    return subsets_dict


def crop_df(df, lat_lims, lon_lims):
    """Crops dataframe to specified lat and lon limits"""
    return df[
        (df.index.get_level_values("latitude") >= min(lat_lims))
        & (df.index.get_level_values("latitude") <= max(lat_lims))
        & (df.index.get_level_values("longitude") >= min(lon_lims))
        & (df.index.get_level_values("longitude") <= max(lon_lims))
    ]


def visualise_train_test_points(
    y_train_df: pd.DataFrame | pd.Series,
    y_test_df: pd.DataFrame | pd.Series,
    do_crop_df: bool = False,
    lat_lims=[-10, -13],
    lon_lims=[147, 149],
    sample_frequency: int = -1,
):
    """
    Visualise the distribution of coral cover values in the train and test sets.

    Args:
        y_train_df (pd.DataFrame | pd.Series): The train set.
        y_test_df (pd.DataFrame | pd.Series): The test set.
        do_crop_df (bool, optional): Whether to crop the dataframes to the specified lat and lon limits.
            Defaults to False.
        lat_lims (list[float], optional): The latitude limits to crop to. Defaults to [-10, -13].
        lon_lims (list[float], optional): The longitude limits to crop to. Defaults to [147, 149].
        sample_frequency (int, optional): The frequency to sample the cropped dataframes at. Defaults to -1.
    """
    # TODO: more customisable plotting
    # combine train and test sets
    comb = pd.concat([y_train_df, y_test_df], axis=1)
    comb["summed"] = comb.fillna(0).sum(axis=1)

    if do_crop_df:
        y_train_plot_df = crop_df(y_train_df, lat_lims, lon_lims)[::sample_frequency]
        y_test_plot_df = crop_df(y_test_df, lat_lims, lon_lims)[::sample_frequency]
        comb_plot_df = crop_df(comb["summed"], lat_lims, lon_lims)[::sample_frequency]

    fig = plt.figure(figsize=(12, 6))
    fig.tight_layout()
    gs = fig.add_gridspec(2, 3)

    plot_titles = ["Train set", "Test set", "All datapoints"]
    hist_colors = ["#3B9AB2", "#EBCC2A", "#d83c04"]
    for i, array in enumerate([y_train_plot_df, y_test_plot_df, comb_plot_df]):
        ax = fig.add_subplot(gs[0, i], projection=ccrs.PlateCarree())
        xa_set = pd_to_array(array)  # create xarray instance
        spatial_plots.plot_spatial(
            xa_set, fax=(fig, ax), cbar=False, title=plot_titles[i]
        )

        ax = fig.add_subplot(gs[1, i])
        ax.set_yscale("log")
        ax.hist(array.values.flatten(), color=hist_colors[i])
        if i == 0:
            ax.set_ylabel("Count (log scale)")

    # superpose train and test sets on all datapoints
    ax.hist(y_train_plot_df.values.flatten(), color=hist_colors[0], alpha=1)
    ax.hist(y_test_plot_df.values.flatten(), color=hist_colors[1], alpha=1)

    # Add a single label to the bottom center of the entire plot
    fig.text(
        0.5,
        0.05,
        "Fraction coral cover",
        ha="center",
        va="center",
        fontsize=12,
        color="black",
    )


def sizeof_fmt(num, suffix="B"):
    """by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified"""
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, "Yi", suffix)


def print_var_mems(num_show):
    for name, size in sorted(
        ((name, sys.getsizeof(value)) for name, value in list(locals().items())),
        key=lambda x: -x[1],
    )[:num_show]:
        print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))


def calculate_class_weight(label_array: np.ndarray):
    unique_values, counts = np.unique(label_array, return_counts=True)
    occurrence_dict = dict(zip(unique_values, counts))
    return occurrence_dict


def fractional_cover_from_reef_check(
    reef_check_df: pd.DataFrame, substrate_codes: list[str] = ["HC", "SC"]
) -> pd.DataFrame:
    """Calculate coral cover from reef check data
    TODO: calculate for each transect section (four per survey)"""
    # number of observations along whole transect
    NUM_SLOTS = 160

    # group by substrate code and survey_id, summing values in "total" column, and
    # keeping all others the same
    reef_check_df_grouped = reef_check_df.groupby(["survey_id", "substrate_code"]).agg(
        {
            "total": "sum",
            **{
                col: "first"
                for col in reef_check_df.columns
                if col not in ["survey_id", "substrate_code", "total"]
            },
        }
    )

    # specify which rows correspond to substrate of interest
    reef_check_df_grouped["is_substrate"] = (
        reef_check_df_grouped.index.get_level_values(1).isin(substrate_codes)
    )
    # # calculate fractional cover
    reef_check_df_grouped["fractional_cover"] = (
        reef_check_df_grouped["total"] / NUM_SLOTS
    )
    # Filter rows based on the "is_substrate" column
    fractional_cover_df = reef_check_df_grouped[reef_check_df_grouped["is_substrate"]]

    # Define the aggregation dictionary
    agg_dict = {
        "fractional_cover": "sum",
        **{
            col: "first"
            for col in reef_check_df_grouped.columns
            if col not in ["fractional_cover", "is_substrate"]
        },
    }

    # Aggregate by "survey_id" using the defined dictionary
    return fractional_cover_df.groupby("survey_id").agg(agg_dict)


# from baselines.py May be useful to calculate magnitude of ocean currents
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

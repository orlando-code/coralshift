# general
import numpy as np
import pandas as pd

# from shapely.geometry import Point

# file handling
import calendar
from tqdm.auto import tqdm
from pathlib import Path

# machine learning
from sklearn.preprocessing import MinMaxScaler, StandardScaler, FunctionTransformer
from sklearn.model_selection import train_test_split
import elapid as ela

# spatial
import xarray as xa
import geopandas as gpd

# custom
# from coralshift.dataloading import get_data
# from coralshift.dataloading.get_data import ReturnRaster, adaptive_depth_mask
from coralshift.dataloading import get_data
from coralshift.utils import utils, config
import cmipper.utils as cmipper_utils


def log_transform(x):
    return np.log(x + 1)


def cont_to_class(array, threshold=0.5):
    """Return thresholded values, leaving nans untouched."""
    return np.where(np.isnan(array), np.nan, np.where(array >= threshold, 1, 0)).astype(
        int
    )


class ProcessMLData:
    """
    Specify datasets, model type, resolution, buffer factor, train/test splits and get back model-ready data
    """

    def __init__(
        self,
        config_info: dict = None,
    ):
        # self.config_info = config_info

        if config_info:
            self.__dict__.update(config_info)
        self.config_info = config_info

    def get_merged_datasets(self):
        dss = []
        datasets = self.datasets
        if not ("gebco" in datasets or "bathymetry" in datasets):
            datasets.append("gebco")

        for dataset in datasets:
            # TODO: other ways to handle this for timeseries i.e. combining static and timeseries
            dss.append(
                get_data.ReturnRaster(
                    lats=self.lats,
                    lons=self.lons,
                    levs=self.levs,
                    resolution=self.resolution,
                    upsample_method=self.upsample_method,
                    downsample_method=self.downsample_method,
                    ds_type=self.ds_type,
                    env_vars=self.env_vars,
                    year_range_to_include=self.year_range_to_include,
                    resolution_unit=self.resolution_unit,
                    config_info=self.config_info,
                ).return_raster(dataset=dataset)
            )

        return xa.merge(dss)

    def return_predictand(self):
        if not self.predictand:  # TODO: better handling of different ways to specify gt
            return "UNEP_GDCR"
        else:
            return self.predictand

    def split_dataset(self, xa_ds):
        df_X, df_y = ds_to_ml_ready(
            xa_ds,
            predictand=self.return_predictand(),
            depth_mask=self.config_info["depth_mask"]
        )
        # if elevation not requested, drop (needed to be included to generate the shallow water mask)
        if not ("gebco" in self.datasets or "bathymetry" in self.datasets):
            print("\nDropping elevation since not a specified covariate")
            df_X.drop(columns="elevation", inplace=True)

        if self.split_type == "pixelwise":
            trains, tests, vals = train_test_val_split(
                df_X,
                df_y,
                ttv_fractions=self.train_test_val_frac,
                split_method=self.split_type,
                random_state=self.random_state,
            )
        elif self.split_type == "checkerboard":
            grid_size = 1  # 1 degree grid  # TODO: make adjustable?
            if grid_size <= self.resolution:
                grid_size *= 2

            merged_df = pd.concat([df_X, df_y], axis=1)
            loose_index_df = merged_df.reset_index()
            # create geometry column from latitude longitude index
            geo_df = gpd.GeoDataFrame(
                loose_index_df,
                geometry=gpd.points_from_xy(
                    loose_index_df.longitude, loose_index_df.latitude
                ),
                crs="EPSG:4326",
            )

            trains_Xy, tests_Xy = ela.checkerboard_split(geo_df, grid_size=grid_size)
            # return geo_df, geo_df, geo_df

            trains = (
                trains_Xy.drop(columns=[self.predictand, "geometry"]).set_index(
                    ["latitude", "longitude"]
                ),
                trains_Xy.drop(columns="geometry").set_index(["latitude", "longitude"])[
                    self.predictand
                ],
            )
            tests = (
                tests_Xy.drop(columns=[self.predictand, "geometry"]).set_index(
                    ["latitude", "longitude"]
                ),
                tests_Xy.drop(columns="geometry").set_index(["latitude", "longitude"])[
                    self.predictand
                ],
            )
            vals = tests
        elif self.split_type == "custom_checkerboard":  # TODO update/standardise this
            X_trains, X_tests = get_specified_ratio_checkerboard_train_test_data(
                df_X, train_test_ratio=self.train_test_val_frac[1])
            y_trains, y_tests = get_specified_ratio_checkerboard_train_test_data(
                df_y, train_test_ratio=self.train_test_val_frac[1])

            trains = (X_trains, y_trains)
            tests = (X_tests, y_tests)
            vals = tests
        elif self.split_type == "spatial":
            X_trains, X_tests = split_train_test_spatial(
               df_X, self.test_geom
            )
            y_trains, y_tests = split_train_test_spatial(
               df_y, self.test_geom
            )
            trains = (X_trains, y_trains)
            tests = (X_tests, y_tests)
            vals = tests

        return trains, tests, vals

    def initialise_data_scaler(self, scaler):
        if scaler == "minmax":
            return MinMaxScaler()
        elif scaler == "standard":
            return StandardScaler()
        elif scaler == "log":   # functioning weirdly
            return FunctionTransformer(log_transform)

    def get_fitted_scaler(self, trains=None, tests=None, vals=None):
        X_scaler = self.initialise_data_scaler(self.X_scaler)
        # fit scaler
        if (trains and tests and vals) is None: #### TODO: WHY?
            trains, tests, vals = self.split_dataset()
        print("\tfitting scaler to X data...")
        return X_scaler.fit(trains[0]), (trains, tests, vals)

    def scale_data(self, trains=None, tests=None, vals=None):
        # TODO: could wrap this in an individual scale function (would involve editing get_fitted_scaler also)
        X_scaler, (trains, tests, vals) = self.get_fitted_scaler(
            trains=trains, tests=tests, vals=vals
        )
        # trains, tests, vals = self.split_dataset()

        if self.y_scaler:
            # if not self.y_scaler:
            y_scaler = self.initialise_data_scaler(self.y_scaler)
            # fit scaler
            print("\tfitting scaler to y data...")
            y_scaler.fit(pd.DataFrame(trains[1]))

        print("\n\ttransforming data...")
        # return appropriately scaled data in format (X_train, y_train), (X_test, y_test), (X_val, y_val
        X_train_scaled = X_scaler.transform(trains[0])
        X_test_scaled = X_scaler.transform(tests[0])
        X_val_scaled = X_scaler.transform(vals[0])

        # return y_scaler

        if self.y_scaler:
            y_train_scaled = y_scaler.transform(pd.DataFrame(trains[1]))
            y_test_scaled = y_scaler.transform(pd.DataFrame(tests[1]))
            y_val_scaled = y_scaler.transform(pd.DataFrame((vals[1])))
        else:
            y_train_scaled = trains[1].to_numpy().flatten()
            y_test_scaled = tests[1].to_numpy().flatten()
            y_val_scaled = vals[1].to_numpy().flatten()

        # return scaled dataframes (now np arrays) as dataframes with their original indices
        return (
            (
                pd.DataFrame(
                    X_train_scaled, index=trains[0].index, columns=trains[0].columns
                ),
                pd.Series(y_train_scaled[:, 0], index=trains[1].index),
            ),
            (
                pd.DataFrame(
                    X_test_scaled, index=tests[0].index, columns=tests[0].columns
                ),
                pd.Series(y_test_scaled[:, 0], index=tests[1].index),
            ),
            (
                pd.DataFrame(
                    X_val_scaled, index=vals[0].index, columns=vals[0].columns
                ),
                pd.Series(y_val_scaled[:, 0], index=vals[1].index),
            ),
        )

    def get_ds_info(self, trains, tests, vals):
        return {
            "class_balance": {
                "train_pos_neg_ratio": float(utils.calc_non_zero_ratio(trains[1], predictand=self.predictand)),
                "test_pos_neg_ratio": float(utils.calc_non_zero_ratio(tests[1], predictand=self.predictand)),
                "val_pos_neg_ratio": float(utils.calc_non_zero_ratio(vals[1], predictand=self.predictand)),
            },
            # TODO: get actual values here somehow
            # TODO: removed this since not always wanting to include elevation
            # "balanced_depth_lims": {
            #     "train": [
            #         min(trains[0]["elevation"]),
            #         max(trains[0]["elevation"]),
            #     ],
            #     "test": [
            #         min(tests[0]["elevation"]),
            #         max(tests[0]["elevation"]),
            #     ],
            #     "val": [
            #         min(vals[0]["elevation"]),
            #         max(vals[0]["elevation"]),
            #     ],
            # },
        }

    def generate_ml_ready_data_from_files(self):

        if self.config_info["data_source"] == "parquet":
            print("Loading preprocessed data from parquet files...")
            res_str = utils.replace_dot_with_dash(str(round(self.config_info["resolution"], 3)))
            spatial_extent_info = cmipper_utils.lat_lon_string_from_tuples(
                self.config_info["lats"], self.config_info["lons"]).upper()

            if self.config_info["depth_mask"]:
                dm = self.config_info["depth_mask"]
                fp_root = Path(config.ml_ready_dir) / self.split_type / f"{str(min(dm))}_{str(max(dm))}"
            else:
                fp_root = Path(config.ml_ready_dir)
            tr_X_fp = fp_root / res_str / f"train_X_{res_str}_{spatial_extent_info}.parquet"
            tr_y_fp = fp_root / res_str / f"train_y_{res_str}_{spatial_extent_info}.parquet"
            te_X_fp = fp_root / res_str / f"test_X_{res_str}_{spatial_extent_info}.parquet"
            te_y_fp = fp_root / res_str / f"test_y_{res_str}_{spatial_extent_info}.parquet"

            # read in train data
            X_trains = pd.read_parquet(tr_X_fp)
            y_trains = pd.read_parquet(tr_y_fp)[0]  # casting back to series
            X_tests = pd.read_parquet(te_X_fp)
            y_tests = pd.read_parquet(te_y_fp)[0]  # casting back to series
        elif self.config_info["data_source"] == "xarray":
            print("TODO")

            # # get merged datasets
            # ds = xa.open_mfdataset(Path(dp).rglob("*.nc"))
            # # split and scale dataset
            # # return ds
            # trains, tests, vals = self.split_dataset(ds)
            # ds_info = self.get_ds_info(trains, tests, vals)
            # # scale data
            # return self.scale_data(trains, tests, vals), ds_info

        return (X_trains, y_trains), (X_tests, y_tests), (X_tests, y_tests), "ds_info"

    def generate_ml_ready_data(self, ds=None):

        if self.config_info["data_source"] in ["parquet", "xarray"]:
            trains, tests, vals, _ = self.generate_ml_ready_data_from_files()
        else:
            if not ds:
                # get merged datasets
                ds = self.get_merged_datasets()
            # split and scale dataset
            trains, tests, vals = self.split_dataset(ds)
        ds_info = self.get_ds_info(trains, tests, vals)
        # scale data
        return self.scale_data(trains, tests, vals), ds_info
        # return trains, tests, vals, ds_info


def ds_to_ml_ready(
    xa_ds: xa.Dataset,
    predictand: str = "UNEP_GDCR",
    target_pos_neg_ratio: float = 0.1,
    initial_depth_mask_lims: tuple[float, float] = [0, 10],
    exclude_list: list[str] = [
        "latitude",
        "longitude",
        "latitude_grid",
        "longitude_grid",
        "crs",
        "depth",
        "spatial_ref",
    ],
    remove_rows: bool = True,
    depth_mask: bool = [-100, 25],
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Convert an xarray Dataset to a format suitable for machine learning.

    Args:
        xa_ds (xa.Dataset): The xarray Dataset to convert.
        predictand (str, optional): The name of the ground truth variable. Defaults to "UNEP_GDCR".
        pos_neg_ratio (float, optional): The ratio of positive to negative samples for classification. Defaults to 0.1.
        depth_mask_lims (tuple[float, float], optional): The depth limits to use for masking. Defaults to [-50, 0].
        exclude_list (list[str], optional): List of variables to exclude from the conversion.
            Defaults to ["latitude", "longitude", "latitude_grid", "longitude_grid", "crs", "depth", "spatial_ref"].
        remove_rows (bool, optional): Whether to remove rows beyond depth limits. Defaults to True.

    Returns:
        tuple[pd.DataFrame, pd.Series]: The converted features (X) and target variable (y).
    """
    # check for predictand and depth variable
    if predictand not in xa_ds:
        raise ValueError("Predictand variable not found in xarray dataset")
    if "elevation" not in xa_ds:
        raise ValueError("Depth variable not found in xarray dataset")

    predictors = [
        pred for pred in xa_ds.variables if pred != predictand and pred not in exclude_list
    ]

    if isinstance(depth_mask, list):
        xa_masked = xa_ds.where((xa_ds["elevation"] >= depth_mask[0]) & (xa_ds["elevation"] <= depth_mask[1]))
    elif depth_mask == "adaptive":
        xa_masked = adaptive_xarray_depth_mask(
            xa_ds,
            initial_depth_mask_lims=initial_depth_mask_lims,
            predictand=predictand,
            depth_var="elevation",
        )
    else:
        depth_mask = xa_ds

    df = xa_masked.to_dataframe()
    # remove nans
    df_nanned = onehot_nan(df, discard_nanrows=remove_rows)
    return df_nanned[predictors], df_nanned[predictand]


def adaptive_xarray_depth_mask(
    xa_ds,
    initial_depth_mask_lims=[0, 10],
    predictand="UNEP_GDCR",
    depth_var="elevation",
    tolerance=0.0001  # Tolerance for detecting significant changes in ratio
):
    """
    Adaptively mask an xarray dataset based on depth, aiming to maximize the positive/negative ratio.
    """
    lower_limit, upper_limit = initial_depth_mask_lims
    step_size = 10  # Initial step size for decreasing the lower limit
    max_iterations = 1000  # Maximum number of iterations to prevent infinite loops
    no_change_limit = 20  # Number of iterations to wait before stopping if no significant change
    iteration = 0
    last_ratios = []

    while iteration < max_iterations:
        # Create the mask based on current limits
        elevation_mask = (xa_ds[depth_var] >= lower_limit) & (xa_ds[depth_var] <= upper_limit)
        masked = xa_ds[predictand].where(elevation_mask, np.nan)    # limit to predictand to save compute

        # Calculate the ratio of non-zero, non-NaN values in the predictand
        num_non_zero = np.count_nonzero(~np.isnan(masked.values) & (masked.values != 0))
        total_points = masked.sizes["latitude"] * masked.sizes["longitude"]
        current_ratio = num_non_zero / total_points

        # Print the current ratio for debugging purposes
        # print(f"Iteration {iteration}: lower_limit={lower_limit}, current_ratio={current_ratio:.4f}")

        # Track changes in ratio
        last_ratios.append(current_ratio)
        if len(last_ratios) > no_change_limit:
            last_ratios.pop(0)

        # Check if there is no significant change in the ratio
        if len(last_ratios) == no_change_limit and max(last_ratios) - min(last_ratios) < tolerance:
            # print("No significant change detected, stopping search.")
            return xa_ds.where(elevation_mask, np.nan)

        # Decrease the lower limit
        lower_limit -= step_size

        iteration += 1

    # If the loop completes without finding a significant maximum, return the last masked dataset
    # print(f"Maximum iterations reached. Final ratio: {current_ratio:.4f}")
    return xa_ds.where(elevation_mask, np.nan)


def calculate_test_cell_buffer(
    test_cell_size: float = 1,
    test_train_ratio: float = 0.2
):
    return np.sqrt(test_train_ratio/2) * test_cell_size


def get_checkerboard_train_test_data(df, test_cell_size: float = 1, test_train_ratio: float = 0.2, initialisation: tuple[float] = (0,0)):
    # calculate necessary buffer for train-test ratio
    buffer = calculate_test_cell_buffer(test_cell_size, test_train_ratio)

    # get min and max lat/lon values from df
    extremes = utils.get_multiindex_min_max(df)
    lat_min, lat_max = extremes['latitude']['min'], extremes['latitude']['max']
    lon_min, lon_max = extremes['longitude']['min'], extremes['longitude']['max']

    # Create a meshgrid of the lat/lon centers
    lat_centres = np.arange(lat_min-initialisation[0], lat_max+test_cell_size, test_cell_size)
    lon_centres = np.arange(lon_min-initialisation[1], lon_max+test_cell_size, test_cell_size)

    # Create a meshgrid of latitudes and longitudes
    lon_grid, lat_grid = np.meshgrid(lon_centres, lat_centres)

    # Create index grids for latitudes and longitudes
    idx_grid = np.arange(lon_grid.shape[1])
    idy_grid = np.arange(lat_grid.shape[0])

    # Use broadcasting to create a full grid of index pairs
    idx_grid, idy_grid = np.meshgrid(idx_grid, idy_grid)

    # Apply the condition to select valid points
    mask = ((idy_grid % 2 == 0) & (idx_grid % 2 == 1)) | ((idy_grid % 2 != 0) & (idx_grid % 2 != 1))

    # Select the latitude and longitude pairs that satisfy the condition
    lat_test_centers = lat_grid[mask]
    lon_test_centers = lon_grid[mask]
    selected_pairs = np.vstack([lon_test_centers, lat_test_centers]).T

    lat_test_regions = np.vstack([lat_test_centers - buffer, lat_test_centers + buffer]).T
    lon_test_regions = np.vstack([lon_test_centers - buffer, lon_test_centers + buffer]).T
    
    test_mask = np.zeros(len(df), dtype=bool)

    for (lat_min_val, lat_max_val), (lon_min_val, lon_max_val) in tqdm(zip(lat_test_regions, lon_test_regions), total=len(lat_test_regions)):
        # iteratively apply the test region mask to dataframe
        combined_mask = (
            (df.index.get_level_values('latitude') >= lat_min_val) & 
            (df.index.get_level_values('latitude') < lat_max_val) &
            (df.index.get_level_values('longitude') >= lon_min_val) & 
            (df.index.get_level_values('longitude') < lon_max_val)
        )
        test_mask |= combined_mask  # Add the current region's mask to the overall test mask

    test = df[test_mask]
    train = df[~test_mask]
    return train, test


def scan_checkerboard(df, test_cell_size: float = 10, test_train_ratio: float = 0.2):

    lat_shifts = np.arange(0, 2*test_cell_size, 2*buffer) 
    lon_shifts = np.arange(0, test_cell_size, 2*buffer) 

    all_tests = []
    for i in lat_shifts:
        for j in lon_shifts:
            # vertical scan
            _, tests = ml_processing.get_checkerboard_train_test_data(data_df, test_cell_size, test_train_ratio, initialisation=(i,j))
            # TODO: perform ML here
            all_tests.append(tests)
            tests_df = pd.concat(all_tests, axis=1).mean(axis=1).rename("value")
    return tests_df

def old_get_checkerboard_train_test_data(df, test_cell_size: float = 1, buffer: float = 0):
    extremes = utils.get_multiindex_min_max(df)

    lat_min, lat_max = extremes['latitude']['min'], extremes['latitude']['max']
    lon_min, lon_max = extremes['longitude']['min'], extremes['longitude']['max']

    lat_centres = np.arange(lat_min, lat_max, test_cell_size)
    lon_centres = np.arange(lon_min, lon_max, test_cell_size)

    # initialise with random seed (and append to list to ensure not repeating).
    # Perhaps check that centre isn't overlapping too much with the previous centres
    check_lat_centres = []
    check_lon_centres = []
    for idy, y0 in enumerate(lat_centres):
        # offset = 0 if idy % 2 == 0 else 1
        for idx, x0 in enumerate(lon_centres):
            if (idy % 2 == 0 and idx % 2 == 1) or (idy % 2 != 0 and idx % 2 != 1):
                check_lat_centres.append(y0)
                check_lon_centres.append(x0)

    # buffer = test_cell_size / 2

    # get the data within the test cell size around each centre
    lat_vals = [(lat_centre - buffer, lat_centre + buffer) for lat_centre in check_lat_centres]
    lon_vals = [(lon_centre - buffer, lon_centre + buffer) for lon_centre in check_lon_centres]

    if isinstance(df, pd.Series):
        df = df.to_frame()
    test_dfs = [df.sort_index().loc[(slice(*lat_val), slice(*lon_val)), :] for lat_val, lon_val in tqdm(
        zip(lat_vals, lon_vals), total=len(lat_vals), desc="compiling test data")]
    test = pd.concat(test_dfs)
    train = df.drop(test.index)

    return train, test


# def get_specified_ratio_checkerboard_train_test_data(
#     df: pd.DataFrame,
#     train_test_ratio: float = 0.2,
# ):
#     # 1 to 1 is 0.5
#     # need areas of trains to be 



# def get_specified_ratio_checkerboard_train_test_data(
#     df: pd.DataFrame,
#     resolution: float = 1,
#     train_test_ratio: float = 0.2,
#     tolerance: float = 0.05,
#     max_count: int = 10
# ):
    # # TODO: implement direction switch checker with update of buffer change. Put into a general search function.
    # # TODO: check that this actually adjusts anything
    # train, test = get_checkerboard_train_test_data(df)
    # df_pos_neg_ratio = len(test) / (len(test)+len(train))

    # counter = 0
    # buffer = resolution / 2
    # while abs(df_pos_neg_ratio - train_test_ratio) > tolerance:
    #     if counter > max_count:
    #         print(
    #             f"Ratio {train_test_ratio} not reached in {max_count} iterations. Final ratio: {df_pos_neg_ratio}")
    #         break
    #     counter += 1
    #     if df_pos_neg_ratio > train_test_ratio:
    #         buffer -= resolution
    #     else:
    #         buffer += resolution

    #     train, test = get_checkerboard_train_test_data(df, buffer=buffer)
    #     df_pos_neg_ratio = len(test) / (len(test)+len(train))
    #     print('ratio': df_pos_neg_ratio)
    # print(f"Final ratio test fraction: {df_pos_neg_ratio}")
    # return train, test


def split_train_test_spatial(df: pd.DataFrame, test_geom: list[float]):
    """Specify a geometry to split the data into train and test sets.
    Geometry should have form [lat_min, lat_max, lon_min, lon_max]."""

    lat_min, lat_max = test_geom[0], test_geom[1]
    lon_min, lon_max = test_geom[2], test_geom[3]

    inner_df = df[
        (df.index.get_level_values("longitude") >= lon_min) & (df.index.get_level_values("longitude") <= lon_max)
        & (df.index.get_level_values("latitude") >= lat_min) & (df.index.get_level_values("latitude") <= lat_max)]

    outer_df = df.drop(inner_df.index)  # inplace is false by default
    return outer_df, inner_df

# TODO: should this be separated out into multiple functions?
# def ds_to_ml_ready(
#     xa_ds: xa.Dataset,
#     predictand: str = "UNEP_GDCR",
#     pos_neg_ratio: float = 0.1,
#     depth_mask_lims: tuple[float, float] = [-50, 0],
#     exclude_list: list[str] = [
#         "latitude",
#         "longitude",
#         "latitude_grid",
#         "longitude_grid",
#         "crs",
#         "depth",
#         "spatial_ref",
#     ],
#     remove_rows: bool = True,
# ) -> tuple[pd.DataFrame, pd.Series]:
#     """
#     Convert an xarray Dataset to a format suitable for machine learning.

#     Args:
#         xa_ds (xa.Dataset): The xarray Dataset to convert.
#         predictand (str, optional): The name of the ground truth variable. Defaults to "UNEP_GDCR".
#         pos_neg_ratio (float, optional): The ratio of positive to negative samples for classification.
#   Defaults to 0.1.
#         depth_mask_lims (tuple[float, float], optional): The depth limits to use for masking. Defaults to [-50, 0].
#         exclude_list (list[str], optional): List of variables to exclude from the conversion.
#             Defaults to ["latitude", "longitude", "latitude_grid", "longitude_grid", "crs", "depth", "spatial_ref"].
#         remove_rows (bool, optional): Whether to remove rows beyond depth limits. Defaults to True.

#     Returns:
#         tuple[pd.DataFrame, pd.Series]: The converted features (X) and target variable (y).
#     """
#     # de-dask and convert to dataframe
#     df = xa_ds.compute().to_dataframe()
#     # checking for empty dfs
#     if len(df) == 0:
#         raise ValueError("Empty dataframe returned from xarray dataset")

#     df["latitude"] = df.index.get_level_values("latitude").round(5)
#     df["longitude"] = df.index.get_level_values("longitude").round(5)

#     # assign rounded values to multiindex of df
#     df = df.set_index(["latitude", "longitude"])

#     predictors = [
#         pred for pred in df.columns if pred != predictand and pred not in exclude_list
#     ]

#     df_masked = get_data.adaptive_depth_mask(
#         df,
#         depth_mask_lims=depth_mask_lims,
#         pos_neg_ratio=pos_neg_ratio,
#         remove_rows=remove_rows,
#         predictand=predictand,
#         depth_var="elevation",
#     )

#     df_nanned = onehot_nan(df_masked, discard_nanrows=remove_rows)
#     # # encode any rows containing nans to additional column
#     # df_masked["nan_onehot"] = df_masked.isna().any(axis=1).astype(int)
#     # # fill any nans with zeros
#     # df_masked = df_masked.fillna(0)

#     X = df_nanned[predictors]
#     y = df_nanned[predictand]

#     return X, y


def onehot_nan(df, discard_nanrows: bool = True):
    """
    One-hot encode the nan values in the data.

    TODO: should I just remove these rows outright?
    """
    nan_rows = df.isna().any(axis=1)

    if discard_nanrows:
        return df.dropna()
    else:
        # encode any rows containing nans to additional column
        df["nan_onehot"] = nan_rows.astype(int)
        # fill any nans with zeros
        return df.fillna(0)


def train_test_val_split(
    df_X: pd.DataFrame,
    df_y: pd.Series,
    ttv_fractions: list[float],
    split_method: str = "pixelwise",
    orientation: str = "vertical",
    random_state: int = 42,
):
    if split_method == "pixelwise":
        train_X, test_val_X, train_y, test_val_y = train_test_split(
            df_X, df_y, test_size=sum(ttv_fractions[1:]), random_state=random_state
        )
        test_size = ttv_fractions[1] / sum(ttv_fractions[1:])
        if test_size == 1:
            test_X, test_y = test_val_X, test_val_y
            # TODO: make less hacky (currently just copying over the same values), even tho val specified as 0
            val_X, val_y = test_val_X, test_val_y
        else:
            test_X, val_X, test_y, val_y = train_test_split(
                test_val_X, test_val_y, test_size=test_size, random_state=random_state
            )
        return (
            (train_X, train_y),
            (test_X, test_y),
            (val_X, val_y),
        )
    elif split_method == "spatial":
        # TODO: correct/check
        return ttv_spatial_split(
            pd.concat([df_X, df_y], axis=1), ttv_fractions, orientiation=orientation
        )


def ttv_spatial_split(
    df: pd.DataFrame, ttv_fractions: list[float], orientation: str = "vertical"
) -> list[pd.DataFrame]:
    """
    Splits a dataframe into train, test, and validation sets, spatially.

    Args:
        df (pd.DataFrame): The dataframe to split.
        ttv_fractions (list[float]): The fractions of the dataframe to allocate to train, test, and validation sets.
        orientation (str, optional): The orientation of the splits. Defaults to "vertical".

    Returns:
        list[pd.DataFrame]: The train, test, and validation sets.
    """
    # check that ttv_fractions sum to 1 or if any is zero
    assert (
        sum(ttv_fractions) == 1 or 0 in ttv_fractions
    ), f"ttv_fractions must sum to 1 or contain a zero. Currently sum to {sum(ttv_fractions)}"

    assert orientation in [
        "vertical",
        "horizontal",
    ], f"orientation must be 'vertical' or 'horizontal'. Currently {orientation}"

    if orientation == "horizontal":
        df = df.sort_values(by="latitude")
    elif orientation == "vertical":
        df = df.sort_values(by="longitude")

    df_list = []

    # this was hanlding omission of val completely: for now, keeping it in, but just empty
    # if 0 in ttv_fractions:
    #     nonzero_fractions = [frac for frac in ttv_fractions if frac != 0]
    #     split_indices = [int(frac * len(df)) for frac in np.cumsum(nonzero_fractions)]

    #     for idx, split_idx in enumerate(split_indices):
    #         if idx == 0:
    #             df_list.append(df.iloc[:split_idx])
    #         elif idx == len(split_indices) - 1:
    #             df_list.append(df.iloc[split_idx:])
    #         else:
    #             df_list.append(df.iloc[split_indices[idx - 1] : split_idx])
    # else:
    df_list = np.split(
        df,
        [
            int(ttv_fractions[0] * len(df)),
            int((ttv_fractions[0] + ttv_fractions[1]) * len(df)),
        ],
    )

    return df_list


def calculate_statistics(
    xa_ds: xa.Dataset,
    vars: list[str] = ["so", "thetao", "tos", "uo", "vo"],
    years_window: tuple[int] = None,
) -> xa.Dataset:
    """
    # TODO:
    Calculate statistics for each variable in the dataset, similar to Couce (2012, 2023).

    Args:
        xa_ds (xa.Dataset): Input xarray dataset.
        vars (list[str], optional): List of variable names to calculate statistics for.
        Defaults to ["so", "thetao", "tos", "uo", "vo"].
        years_window (tuple[int], optional): The time period to calculate statistics for. Defaults to None.

    Returns:
        xa.Dataset: Dataset containing the calculated statistics.
    """
    if years_window:
        # Select the time period of interest
        xa_ds = xa_ds.sel(
            time=slice(
                utils.year_to_datetime(min(years_window)),
                utils.year_to_datetime(max(years_window)),
            )
        )

    stats = {}

    for i, var_name in tqdm(
        enumerate(vars), desc="calculating statistics for variables", total=len(vars)
    ):
        var_data = xa_ds[var_name]
        # Calculate annual average
        # annual_mean = var_data.resample(time='1Y').mean()
        # stats[f"{var_name}_am"] = annual_mean

        # Calculate mean for each month
        monthly_mean = var_data.groupby("time.month").mean(dim="time")

        # Map numerical month values to month names
        month_names = [calendar.month_name[i] for i in monthly_mean["month"].values]

        # Assign monthly means to their respective month names
        for i, month in enumerate(month_names):
            stats[f"{var_name}_{month.lower()}_mean"] = monthly_mean.isel(
                month=i
            ).values

        # Calculate maximum and minimum of monthly values over the whole time period
        monthly_max_overall = var_data.groupby("time.month").max(dim="time")
        monthly_min_overall = var_data.groupby("time.month").min(dim="time")

        for i, month in enumerate(month_names):
            stats[f"{var_name}_{month.lower()}_max"] = monthly_max_overall.isel(
                month=i
            ).values
            stats[f"{var_name}_{month.lower()}_min"] = monthly_min_overall.isel(
                month=i
            ).values

        # Calculate standard deviation of time steps
        time_std = var_data.std(dim="time", skipna=None)
        stats[f"{var_name}_time_std"] = time_std.values

        # Calculate standard deviation of January and July values
        january_std = var_data.where(var_data["time.month"] == 1).std(dim="time")
        july_std = var_data.where(var_data["time.month"] == 7).std(dim="time")
        stats[f"{var_name}_jan_std"] = january_std.values
        stats[f"{var_name}_jul_std"] = july_std.values

        # Calculate the overall mean for each statistic
        stats[f"{var_name}_overall_mean"] = var_data.mean(dim="time").values

    # Combine all calculated variables into a new dataset, retaining the original dataset's attributes, coordinates etc.
    # stats_xa = xa.Dataset(stats)

    stats_xa = xa.Dataset(
        {key: (("latitude", "longitude"), value) for key, value in stats.items()},
        coords={"latitude": var_data.latitude, "longitude": var_data.longitude},
    )

    stats_xa.attrs = xa_ds.attrs

    return stats_xa

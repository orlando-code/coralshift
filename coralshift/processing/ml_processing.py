# general
import numpy as np
import pandas as pd

# file handling
import calendar
from tqdm.auto import tqdm

# machine learning
from sklearn.preprocessing import MinMaxScaler, StandardScaler, FunctionTransformer
from sklearn.model_selection import train_test_split

# spatial
import xarray as xa

# custom
# from coralshift.dataloading import get_data
# from coralshift.dataloading.get_data import ReturnRaster, adaptive_depth_mask
from coralshift.dataloading import get_data
from coralshift.utils import utils


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
        model_data: str = "static",
        datasets: list[str] = ["cmip6", "unep", "gebco"],
        train_test_val_frac: list = [0.8, 0.2, 0],
        X_scaler: str = "minmax",
        y_scaler: str = "log",
        split_type: str = "pixelwise",
        depth_mask_lims: list[float, float] = [
            -50,
            0,
        ],  # TODO: user-specified or calculated as a fraction of class imbalance?
        lats: list[float, float] = [-40, 0],
        lons: list[float, float] = [130, 170],
        levs: list[int, int] = [0, 20],
        resolution: float = 1,
        pos_neg_ratio: float = 0.1,
        resolution_unit: str = "d",
        upsample_method: str = "linear",
        downsample_method: str = "mean",
        env_vars: list[str] = [
            "rsdo",
            # "mlotst", "so", "thetao", "uo", "vo", "tos"
        ],
        predictand: str = "UNEP_GDCR",
        year_range_to_include: list[int, int] = [1950, 2014],
        random_state: int = 42,
        config_info: dict = None,
    ):
        self.model_data = model_data
        self.datasets = datasets
        self.train_test_val_frac = train_test_val_frac
        self.X_scaler = X_scaler
        self.y_scaler = y_scaler
        self.split_type = split_type
        self.depth_mask_lims = depth_mask_lims
        self.lats = lats
        self.lons = lons
        self.levs = levs
        self.resolution = resolution
        self.resolution_unit = resolution_unit
        self.upsample_method = upsample_method
        self.downsample_method = downsample_method
        self.env_vars = env_vars
        self.predictand = predictand
        self.year_range_to_include = year_range_to_include
        self.random_state = random_state
        self.config_info = config_info
        self.pos_neg_ratio = pos_neg_ratio

        if config_info:
            self.__dict__.update(config_info)

    def get_merged_datasets(self):
        dss = []
        for dataset in self.datasets:
            # TODO: other ways to handle this for timeseries i.e. combining static and timeseries
            dss.append(
                get_data.ReturnRaster(
                    dataset=dataset,
                    lats=self.lats,
                    lons=self.lons,
                    levs=self.levs,
                    resolution=self.resolution,
                    upsample_method=self.upsample_method,
                    downsample_method=self.downsample_method,
                    ds_type=self.model_data,
                    env_vars=self.env_vars,
                    year_range_to_include=self.year_range_to_include,
                    resolution_unit=self.resolution_unit,
                    config_info=self.config_info,
                ).return_raster()
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
            pos_neg_ratio=self.pos_neg_ratio,
            depth_mask_lims=self.depth_mask_lims,
        )

        trains, tests, vals = train_test_val_split(
            df_X,
            df_y,
            ttv_fractions=self.train_test_val_frac,
            split_method=self.split_type,
            random_state=self.random_state,
        )

        return trains, tests, vals

    def initialise_data_scaler(self):
        scaler_type = self.X_scaler
        if scaler_type == "minmax":
            return MinMaxScaler()
        elif scaler_type == "standard":
            return StandardScaler()
        elif scaler_type == "log":
            return FunctionTransformer(log_transform)

    def get_fitted_scaler(self, trains=None, tests=None, vals=None):
        X_scaler = self.initialise_data_scaler()
        # y_scaler = self.initialise_data_scaler(self.y_scaler)
        # fit scaler
        if (trains and tests and vals) is None:
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
            y_scaler = self.initialise_data_scaler()
            # fit scaler
            print("\tfitting scaler to y data...")
            y_scaler.fit(pd.DataFrame(trains[1]))

        print("\n\ttransforming data...")
        # return appropriately scaled data in format (X_train, y_train), (X_test, y_test), (X_val, y_val
        X_train_scaled = X_scaler.transform(trains[0])
        X_test_scaled = X_scaler.transform(tests[0])
        X_val_scaled = X_scaler.transform(vals[0])

        y_train_scaled = y_scaler.transform(
            pd.DataFrame(trains[1]) if y_scaler else pd.DataFrame(trains[1])
        )
        y_test_scaled = y_scaler.transform(
            pd.DataFrame(tests[1]) if y_scaler else pd.DataFrame(tests[1])
        )
        y_val_scaled = y_scaler.transform(
            pd.DataFrame(vals[1]) if y_scaler else pd.DataFrame(vals[1])
        )

        # return scaled dataframes (now np arrays) as dataframes with their original indices
        return (
            (
                pd.DataFrame(
                    X_train_scaled, index=trains[0].index, columns=trains[0].columns
                ),
                pd.Series(y_train_scaled.flatten(), index=trains[1].index),
            ),
            (
                pd.DataFrame(
                    X_test_scaled, index=tests[0].index, columns=tests[0].columns
                ),
                pd.Series(y_test_scaled.flatten(), index=tests[1].index),
            ),
            (
                pd.DataFrame(
                    X_val_scaled, index=vals[0].index, columns=vals[0].columns
                ),
                pd.Series(y_val_scaled.flatten(), index=vals[1].index),
            ),
        )

    def get_ds_info(self, trains, tests, vals):
        return {
            "class_balance": {
                "train_pos_neg_ratio": float(utils.calc_non_zero_ratio(trains[1])),
                "test_pos_neg_ratio": float(utils.calc_non_zero_ratio(tests[1])),
                "val_pos_neg_ratio": float(utils.calc_non_zero_ratio(vals[1])),
            },
            # TODO: get actual values here somehow
            "balanced_depth_lims": {
                "train": [
                    min(trains[0]["elevation"]),
                    max(trains[0]["elevation"]),
                ],
                "test": [
                    min(tests[0]["elevation"]),
                    max(tests[0]["elevation"]),
                ],
                "val": [
                    min(vals[0]["elevation"]),
                    max(vals[0]["elevation"]),
                ],
            },
        }

    def generate_ml_ready_data(self):
        # get merged datasets
        ds = self.get_merged_datasets()
        # split and scale dataset
        trains, tests, vals = self.split_dataset(ds)
        ds_info = self.get_ds_info(trains, tests, vals)
        # scale data
        return self.scale_data(trains, tests, vals), ds_info


# TODO: should this be separated out into multiple functions?
def ds_to_ml_ready(
    xa_ds: xa.Dataset,
    predictand: str = "UNEP_GDCR",
    pos_neg_ratio: float = 0.1,
    depth_mask_lims: tuple[float, float] = [-50, 0],
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
    # de-dask and convert to dataframe
    df = xa_ds.compute().to_dataframe()
    # TODO: implement checking for empty dfs

    predictors = [
        pred for pred in df.columns if pred != predictand and pred not in exclude_list
    ]

    df_masked = get_data.adaptive_depth_mask(
        df,
        depth_mask_lims=depth_mask_lims,
        pos_neg_ratio=pos_neg_ratio,
        remove_rows=remove_rows,
        predictand=predictand,
        depth_var="elevation",
    )

    df_nanned = onehot_nan(df_masked, discard_nanrows=remove_rows)
    # # encode any rows containing nans to additional column
    # df_masked["nan_onehot"] = df_masked.isna().any(axis=1).astype(int)
    # # fill any nans with zeros
    # df_masked = df_masked.fillna(0)

    X = df_nanned[predictors]
    y = df_nanned[predictand]

    return X, y


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

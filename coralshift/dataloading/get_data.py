# TODO: automate with nice data configs of selected variables etc.
# TODO: logging functions

# import xarray as xa
# from coralshift.dataloading import config
# from coralshift import functions_creche
# from coralshift.processing import spatial_data
# from coralshift.machine_learning import baselines
# from coralshift import functions_creche
# from pathlib import Path
# import geopandas as gpd

import dask_geopandas as daskgpd
import shapely.geometry as sgeometry
import matplotlib.pyplot as plt
import cartopy.crs as ccrs


# import dask.dataframe as dd
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler


# compute/file handling
# import multiprocessing
from functools import lru_cache

import subprocess
from concurrent.futures import ThreadPoolExecutor
import joblib

# from dask.distributed import Client, LocalCluster
# import dask
# import pickle

import time
from tqdm.auto import tqdm

# import time
# import random
# from tqdm import tqdm
from pathlib import Path

# ml
# import xgboost as xgb
# from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    FunctionTransformer,
    RobustScaler,
    PowerTransformer,
    QuantileTransformer,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import sklearn.metrics as sklmetrics

import xgboost as xgb


# from sklearn.metrics import root_mean_squared_error, mean_absolute_error

# custom
from coralshift import functions_creche
from coralshift.dataloading import config, bathymetry
from coralshift.utils import utils, file_ops
from coralshift.processing import spatial_data
from coralshift.plotting import model_results, spatial_plots

# general
import numpy as np
import pandas as pd
import xarray as xa

# COMPUTE SETUP
FRAC_COMPUTE = 0.25
CV_FOLDS = 3
N_ITER = 10
EARLY_STOPPING_ROUNDS = 10
EVAL_METRIC = "mae"
NUM_BOOST_ROUND = 100

# DATA SETUP
STATIC_DATA_DIR_FP = Path(config.static_cmip6_data_folder)
DEPTH_MASK_LIMS = [-50, 0]
GROUND_TRUTH_NAME = "unep_coral_presence"
RES_STR = "0-01"
TTV_FRACTIONS = [0.75, 0.25, 0]
SPLIT_METHOD = "pixelwise"
SCALE_METHOD_X = "minmax"
SCALE_METHOD_y = "log"
REMOVE_NAN_ROWS = True

DATA_SUBSET = 100


# decorator to hold function output in memory
@lru_cache(maxsize=None)  # Set maxsize to limit the cache size, or None for unlimited
def load_static_data(res_str: str = "0-01"):
    # TODO: could replace with path to multifile dir, open with open_mfdataset and chunking
    high_res_ds_fp = STATIC_DATA_DIR_FP.glob(f"*_{res_str}_*.nc").__next__()
    high_res_ds = xa.open_dataset(high_res_ds_fp)
    all_data_df = high_res_ds.to_dataframe()

    return all_data_df


def split_X_y(df, target_name: str):
    y = df[target_name]
    X = df.drop(target_name, axis=1)

    return X, y


def generate_data_scaler(method: str = "minmax"):
    """
    Scale the data using the specified method.
    """
    return {
        "minmax": MinMaxScaler(),
        "standard": StandardScaler(),
        "robust": RobustScaler(),
        "quantile": QuantileTransformer(),
        "box-cox": PowerTransformer(method="box-cox"),
        "yeo-johnson": PowerTransformer(method="yeo-johnson"),
        "log": FunctionTransformer(log_transform),
    }.get(method)


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


def log_transform(x):
    return np.log(x + 1)


def get_ml_ready_static_data(
    res_str: str = RES_STR,
    depth_mask_lims: list[float, float] = DEPTH_MASK_LIMS,
    ground_truth_name: str = GROUND_TRUTH_NAME,
    static_data_dir_fp: str = STATIC_DATA_DIR_FP,
    ttv_fractions: list[float, float, float] = TTV_FRACTIONS,
    split_method: str = SPLIT_METHOD,
    scale_method_X: str = SCALE_METHOD_X,
    scale_method_y: str = SCALE_METHOD_y,
    remove_nan_rows: bool = REMOVE_NAN_ROWS,
    random_state: int = 42,
    verbosity: int = 0,
):
    # TODO: shift from global to arguments
    # load data. TODO: add multifile loading for machines with reasonable numbers of CPUs
    if verbosity > 0:
        print("loading static data...")
    all_data_df = load_static_data(RES_STR)

    # DATA PREPROCESSING
    # apply depth mask
    if verbosity > 0:
        print("preprocessing...")
    depth_masked_df = functions_creche.depth_filter(
        all_data_df, depth_mask_lims=DEPTH_MASK_LIMS
    )
    # split data into train, test, validation
    [train_df, test_df, val_df] = functions_creche.train_test_val_split(
        depth_masked_df,
        ttv_fractions=TTV_FRACTIONS,
        split_method=SPLIT_METHOD,
        random_state=random_state,
    )
    # preprocess data: initialise scalers values
    scaler_X = generate_data_scaler(SCALE_METHOD_X)
    scaler_y = generate_data_scaler(SCALE_METHOD_y)
    # preprocess data: nan handling
    [train_df, test_df, val_df] = [
        onehot_nan(df, REMOVE_NAN_ROWS) for df in [train_df, test_df, val_df]
    ]
    # split into X and y for each subset
    (train_X, train_y), (test_X, test_y), (val_X, val_y) = [
        split_X_y(df, GROUND_TRUTH_NAME) for df in [train_df, test_df, val_df]
    ]
    # fit scalers
    scaler_X.fit(train_X)
    scaler_y.fit(train_y)
    # transform data
    (train_X, train_y), (test_X, test_y), (val_X, val_y) = [
        (scaler_X.transform(X), scaler_y.transform(y))
        for X, y in [(train_X, train_y), (test_X, test_y), (val_X, val_y)]
    ]
    return (train_X, train_y), (test_X, test_y), (val_X, val_y)


def generate_unep_xarray(
    lats: list[float, float] = [-90, 90],
    lons: list[float, float] = [-180, 180],
    degrees_resolution: float = 15 / 3600,
):
    # check if unep xarray already exists
    res_str = utils.replace_dot_with_dash(str(round(degrees_resolution, 3)))

    unep_xa_dir = Path(config.gt_data_dir) / "unep_wcmc/rasters"
    unep_xa_dir.mkdir(parents=True, exist_ok=True)

    spatial_extent_info = functions_creche.lat_lon_string_from_tuples(
        lats, lons
    ).upper()
    unep_xa_fp = unep_xa_dir / f"unep_{res_str}_{spatial_extent_info}.nc"

    if unep_xa_fp.exists():
        print(f"Loading UNEP xarray at {degrees_resolution:.03f} degrees resolution.")
        return xa.open_dataset(unep_xa_fp)
    else:
        print("loading UNEP data...")
        unep_fp = (
            Path(config.gt_data_dir)
            / "unep_wcmc/01_Data/WCMC008_CoralReef2021_Py_v4_1.shp"
        )
        # load unep tabular data. Don't dask yet to allow filtering by region (if required)
        # unep_gdf = gpd.read_file(unep_fp).cx[lats[0] : lats[1], lons[0] : lons[1]]
        unep_gdf = daskgpd.read_file(unep_fp, npartitions=4)
        geometry_filter = sgeometry.box(min(lons), min(lats), max(lons), max(lats))
        filtered_gdf = unep_gdf[unep_gdf.geometry.intersects(geometry_filter)]

        print(
            f"generating UNEP raster at {degrees_resolution:.03f} degrees resolution..."
        )
        # generate gt raster
        # Purist: defined here as the mean (lat/lon) value maximum resolution (30m) the UNEP data at the equator
        # degrees_resolution = spatial_data.distance_to_degrees(
        #     distance_lat=452, approx_lat=0, approx_lon=0
        # )[-1]
        # for now, resolution of global bathymetry
        # generating a raster necessitates a resolution of grid cell.
        unep_raster = functions_creche.rasterize_geodf(
            filtered_gdf, resolution=degrees_resolution
        )

        print("casting raster to xarray...")
        # generate gt xarray
        unep_xa = functions_creche.raster_to_xarray(
            unep_raster,
            x_y_limits=functions_creche.lat_lon_vals_from_geo_df(filtered_gdf)[:4],
            resolution=degrees_resolution,
            name="UNEP_GDCR",
        ).chunk("auto")

        # save to filepath
        print(f"saving UNEP raster to {unep_xa_fp}...")
        unep_xa.to_netcdf(unep_xa_fp)

        return unep_xa.to_dataset()


def generate_reef_check_df_from_csv(substrate_csv_fp: str | Path):
    """
    Load points from a csv file and return a geopandas dataframe.
    """
    # read csv. Specify date format and convert any non-numeric values in "total" column to NaN
    df_substrate = pd.read_csv(
        substrate_csv_fp,
        parse_dates=["date"],
        date_format="%d-%B-%y",
        converters={"total": utils.convert_to_numeric},
    )

    # PREPROCESSING
    # some entries in "total" column are "O" rather than "0"!
    df_substrate.total.replace("O", "0")
    df_substrate["total"] = (
        pd.to_numeric(df_substrate["total"], errors="coerce").fillna(0).astype("int64")
    )
    # Split the column on the comma
    split_series = df_substrate["coordinates_in_decimal_degree_format"].str.split(",")
    # strip any leading/trailing spaces
    split_series = split_series.apply(lambda x: [(val.strip()) for val in x])
    # convert to float if possible
    out = split_series[:].apply(
        lambda x: [functions_creche.try_convert_to_float(val.strip()) for val in x]
    )
    # write to separate columns in df
    df_substrate[["latitude", "longitude"]] = pd.DataFrame(
        out.to_list(), index=df_substrate.index
    )
    # remove all rows containing a nan in either of these two columns and return result
    return df_substrate.dropna(subset=["latitude", "longitude"])


def generate_reef_check_points():

    reef_check_points_fp = Path(config.gt_data_dir) / "reef_check/reef_check_points.pkl"

    if reef_check_points_fp.exists():
        print(f"Reef Check point data at {reef_check_points_fp} already exists.")
        return pd.read_pickle(reef_check_points_fp)
    else:
        print("loading csv...")
        substrate_csv_fp = Path(config.gt_data_dir) / "reef_check/Substrate.csv"
        print("processing csv...")
        reef_check_df = generate_reef_check_df_from_csv(substrate_csv_fp)
        print("saving dataframe to pickle...")  # TODO: better file storage?
        reef_check_df.to_pickle(reef_check_points_fp)

        return reef_check_df


def generate_reef_check_xarray(resolution: float = 0.01):

    res_str = utils.replace_dot_with_dash(str(resolution))

    reef_check_xa_dir = Path(config.gt_data_dir) / "reef_check/rasters"
    reef_check_xa_dir.mkdir(parents=True, exist_ok=True)

    reef_check_xa_fp = reef_check_xa_dir / f"reef_check_{res_str}.nc"

    if reef_check_xa_fp.exists():
        print(f"Reef Check xarray at {res_str} already exists.")
        return xa.open_dataset(reef_check_xa_fp)
    else:
        # generate_reef_check_xarray()
        print("to implement")


def generate_gebco_xarray(
    lats: list[float, float] = [-40, 0],
    lons: list[float, float] = [130, 170],
):
    # TODO: make generic for other bathymetry data?
    # TODO: mosaicing of files
    gebco_xa_dir = Path(config.bathymetry_folder) / "gebco"
    gebco_xa_dir.mkdir(parents=True, exist_ok=True)

    # check if there exists a file spanning an adequate spatial extent
    nc_fps = list(Path(gebco_xa_dir).glob("*.nc"))
    subset_fps = functions_creche.find_files_for_area(nc_fps, lats, lons)
    if len(subset_fps) > 0:
        # arbitrarily select first suitable file
        gebco_fp = subset_fps[0]
    else:
        raise FileNotFoundError(
            "No GEBCO files with suitable geographic range found. Ensure necessary region is downloaded."
        )
    print(
        f"Loading gebco xarray across {lats} latitudes & {lons} longitudes from {gebco_fp}."
    )
    return spatial_data.process_xa_d(xa.open_dataset(gebco_fp)).sel(
        latitude=slice(min(lats), max(lats)), longitude=slice(min(lons), max(lons))
    )  # TODO: proper chunking


def generate_gebco_slopes_xarray(
    lats: list[float, float] = [-40, 0],
    lons: list[float, float] = [130, 170],
):
    gebco_xa_dir = Path(config.bathymetry_folder) / "gebco/rasters"
    gebco_xa_dir.mkdir(parents=True, exist_ok=True)

    # check if there exists a file spanning an adequate spatial extent
    # TODO: ideally need to buffer to allow calculation of edge cases
    nc_fps = list(Path(gebco_xa_dir).glob("*.nc"))
    subset_fps = functions_creche.find_files_for_area(nc_fps, lats, lons)
    # select fps from subset_fps which contain "slope"
    subset_slope_fps = [fp for fp in subset_fps if "slope" in str(fp)]

    if len(subset_slope_fps) > 0:
        # arbitrarily select first suitable file
        print(
            f"Loading seafloor slopes xarray across {lats} latitudes & {lons} longitudes from {subset_slope_fps[0]}."
        )
        return xa.open_dataset(subset_slope_fps[0]).sel(
            latitude=slice(min(lats), max(lats)), longitude=slice(min(lons), max(lons))
        )
    else:
        gebco_xa = generate_gebco_xarray(lats, lons)
        print("calculating slopes from bathymetry...")
        return (
            spatial_data.process_xa_d(
                bathymetry.calculate_gradient_magnitude(gebco_xa["elevation"])
            )
            .to_dataset(name="slope")
            .sel(
                latitude=slice(min(lats), max(lats)),
                longitude=slice(min(lons), max(lons)),
            )
        )


def construct_cmip_command(
    variable: str,
    source_id: str = "EC-Earth3P-HR",
    member_id: str = "r1i1p2f1",
    lats: list[float, float] = [-40, 0],
    lons: list[float, float] = [130, 170],
    levs: list[int, int] = [0, 20],
    script_fp: str = "coralshift/dataloading/download_cmip6_data_parallel.py",
):
    arguments = [
        "--source_id",
        source_id,
        "--member_id",
        member_id,
        "--variable_id",
        variable,
    ]

    # Define the list arguments with their corresponding values
    list_arguments = [
        ("--lats", lats),
        ("--lons", lons),
        ("--levs", levs),
    ]
    list_str = [
        item
        for sublist in [
            [flag, str(val)] for flag, vals in list_arguments for val in vals
        ]
        for item in sublist
    ]

    arguments += list_str
    return ["/home/rt582/miniforge3/envs/coralshift/bin/python", script_fp] + arguments


def execute_subprocess_command(command, output_log_path):
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        print(f"{output_log_path}")
        with open(output_log_path, "w") as output_log_file:
            output_log_file.write(result.stdout.decode())
            output_log_file.write(result.stderr.decode())

        # with open(str(output_log_path), "w") as error_log_file:
        #     error_log_file.write(result.stderr.decode())

    except subprocess.CalledProcessError as e:
        print(f"Error{e}")
        # # Handle the exception if needed
        with open(output_log_path, "w") as error_log_file:
            error_log_file.write(f"Error: {e}")
            error_log_file.write(result.stderr.decode())


def run_commands_with_thread_pool(cmip_commands, output_log_paths, error_log_paths):
    with ThreadPoolExecutor(max_workers=len(cmip_commands)) as executor:
        # Submit each script command to the executor
        executor.map(execute_subprocess_command, cmip_commands, output_log_paths)


def ensure_cmip6_downloaded(
    variables: list[str],
    source_id: str = "EC-Earth3P-HR",
    member_id: str = "r1i1p2f1",
    lats: list[float, float] = [-40, 0],
    lons: list[float, float] = [130, 170],
    year_range_to_include: list[int, int] = [1950, 2014],
    levs: list[int, int] = [0, 20],
    logging_dir: str = "/maps/rt582/coralshift/newtestlogs/cmip6_download_logs",
):
    potential_ds, potential_ds_fp = find_intersecting_cmip(
        variables, source_id, member_id, lats, lons, levs
    )
    if potential_ds is None:
        if not Path(logging_dir).exists():
            Path(logging_dir).mkdir(parents=True, exist_ok=True)

        print("Creating/downloading necessary file(s)...")
        # TODO: add in other sources/members

        cmip_commands, output_log_paths = construct_cmip_commands(
            variables, source_id, member_id, lats, lons, levs
        )
        run_commands_with_thread_pool(cmip_commands, output_log_paths, output_log_paths)
    else:
        print(
            f"CMIP6 file with necessary variables spanning latitudes {lats} and longitudes {lons} already exists at: \n{potential_ds_fp}"  # noqa
        )


def find_intersecting_cmip(
    variables: list[str],
    source_id: str = "EC-Earth3P-HR",
    member_id: str = "r1i1p2f1",
    lats: list[float, float] = [-40, 0],
    lons: list[float, float] = [130, 170],
    year_range_to_include: list[int, int] = [1950, 2014],
    levs: list[int, int] = [0, 20],
):

    # check whether intersecting cropped file already exists
    cmip6_dir_fp = Path(config.cmip6_data_folder) / source_id / member_id
    # TODO: include levs check
    correct_area_fps = list(
        functions_creche.find_files_for_area(
            cmip6_dir_fp.rglob("*.nc"), lat_range=lats, lon_range=lons
        ),
    )
    # TODO: check that also spans full year range
    if len(correct_area_fps) > 0:
        # check that file includes all variables in variables list
        for fp in correct_area_fps:
            if all(variable in str(fp) for variable in variables):
                return (
                    spatial_data.process_xa_d(xa.open_dataset(fp)).sel(
                        latitude=slice(min(lats), max(lats)),
                        longitude=slice(min(lons), max(lons)),
                    ),
                    fp,
                )
    return None, None


def construct_cmip_commands(
    variables,
    source_id: str = "EC-Earth3P-HR",
    member_id: str = "r1i1p2f1",
    lats: list[float, float] = [-40, 0],
    lons: list[float, float] = [130, 170],
    levs: list[int, int] = [0, 20],
    logging_dir: str = "/maps/rt582/coralshift/newtestlogs/cmip6_download_logs",
) -> (list, list):
    cmip_commands = []
    output_log_paths = []
    if not Path(logging_dir).exists():
        Path(logging_dir).mkdir(parents=True, exist_ok=True)

    for variable in variables:
        # if not, run necessary downloading
        cmip_command = construct_cmip_command(
            variable, source_id, member_id, lats, lons, levs
        )
        cmip_commands.append(cmip_command)

        output_log_paths.append(
            Path(logging_dir) / f"{source_id}_{member_id}_{variable}.txt"
        )
    return cmip_commands, output_log_paths


def generate_cmip_raster(
    variables: list[str],
    lats: list[float, float],
    lons: list[float, float],
    levs: list[int, int],
    year_range_to_include: list[int, int],
    source: str = "EC-Earth3P-HR",
    member: str = "r1i1p1f1",
):  # not currrently used

    ensure_cmip6_downloaded(variables, lats=lats, lons=lons, levs=levs)

    fname = functions_creche.FileName(
        variable_id=variables,
        grid_type="latlon",
        fname_type="var_concatted",
        lats=lats,
        lons=lons,
        plevels=levs,
        date_range=[min(year_range_to_include), max(year_range_to_include)],
    ).construct_fname()

    cmip_data_dir = (
        Path(config.cmip6_data_folder) / source / member / "testing"
    )  # TODO: replace testing
    cmip_fp = cmip_data_dir / fname

    # return spatial_data.process_xa_d(xa.open_dataset(cmip_fp))


class ReturnRaster:
    """
    Currently accepted values for "dataset"
    - unep / unep_wcmc / gdcr / unep_coral_presence
    - gebco / bathymetry
    - gebco_slope / bathymetry_slope
    - cmip6
    # - reef_check / reef_check_points

    """

    def __init__(
        self,
        dataset: str,
        lats: list[float, float] = [-90, 90],
        lons: list[float, float] = [-180, 180],
        levs: list[int, int] = [0, 20],
        resolution: float = 1,
        resolution_unit: str = "d",
        pos_neg_ratio: float = 0.1,
        upsample_method: str = "linear",
        downsample_method: str = "mean",
        spatial_buffer: int = 2,
        ds_type: str = None,
        env_vars: list[str] = ["rsdo", "mlotst", "so", "thetao", "uo", "vo", "tos"],
        year_range_to_include: list[int, int] = [1950, 2014],
        source: str = "EC-Earth3P-HR",
        member: str = "r1i1p1f1",
        config_info: dict = None,
    ):
        self.dataset = dataset
        self.lats = lats
        self.lons = lons
        self.levs = levs
        self.resolution = resolution
        self.resolution_unit = resolution_unit
        self.pos_neg_ratio = pos_neg_ratio
        self.upsample_method = upsample_method
        self.downsample_method = downsample_method
        self.spatial_buffer = spatial_buffer
        self.ds_type = ds_type
        self.env_vars = env_vars
        self.year_range_to_include = year_range_to_include
        self.config_info = config_info

        if config_info:
            self.__dict__.update(config_info)

    def get_raw_raster(self):
        if self.dataset in ["unep", "unep_wcmc", "gdcr", "unep_coral_presence"]:
            return generate_unep_xarray(self.lats, self.lons)
        elif self.dataset in ["gebco", "bathymetry"]:
            return generate_gebco_xarray(self.lats, self.lons)
        elif self.dataset in ["gebco_slope", "bathymetry_slope"]:
            return generate_gebco_slopes_xarray(self.lats, self.lons)
        elif self.dataset in ["cmip6", "cmip"]:
            # ensure necessary files downloaded: variables, years, lats, lons, levs
            # TODO: (probably) – split up download and processing, ensuring download for all variables
            # finishes before processing. May also involve changing how variables are passed (i.e. call
            # to processing will take a list of variables and will only be performed once)
            # running into issues with wider area due to [0,-50] tos file apparently being text????
            ensure_cmip6_downloaded(
                variables=self.env_vars,
                lats=self.lats,
                lons=self.lons,
                levs=self.levs,
            )
            # TODO: potentially tidy variable assignment in find_intersecting_cmip
            raster, _ = find_intersecting_cmip(
                self.env_vars, lats=self.lats, lons=self.lons, levs=self.levs
            )

            if self.ds_type == "static":
                raster = self.process_timeseries_to_static(raster)

            return raster

        else:
            raise ValueError(f"Dataset {self.dataset} not recognised.")

    def get_resampled_raster(self, raster):
        self.resolution = spatial_data.process_resolution_input(
            self.resolution, self.resolution_unit
        )
        print(f"\tresampling dataset to {self.resolution} degree(s) resolution...\n")

        current_resolution = functions_creche.get_resolution(raster)
        if self.resolution < current_resolution:
            resample_method = self.upsample_method
        else:
            resample_method = self.downsample_method

        # doing this to get around "buffer size too small error"
        rough_regrid = functions_creche.resample_xa_d(
            raster,
            lat_range=self.lats,
            lon_range=self.lons,
            resolution=self.resolution,
            resample_method=resample_method,
        )
        return functions_creche.xesmf_regrid(
            rough_regrid,
            lat_range=self.lats,
            lon_range=self.lons,
            resolution=self.resolution,
            # resample_method=resample_method,
        )

    def get_spatially_buffered_raster(self, raster):
        print("\tapplying spatial buffering...")
        return functions_creche.apply_fill_loess(
            raster, nx=self.spatial_buffer, ny=self.spatial_buffer
        )

    def process_timeseries_to_static(self, raster):
        # TODO: add processing for other timseries datasets
        if self.dataset in ["cmip6", "cmip"]:
            print("\tcalculating statistics for static ML model(s)...")
            static_ds = functions_creche.calculate_statistics(
                raster,
                vars=self.env_vars,
                years_window=self.year_range_to_include,
            )
        else:
            raise ValueError(
                f"Dataset {self.dataset} not recognised as appropriate timeseries."
            )
        return static_ds

    def return_raster(self):
        # order of operations decided to minimise unnecessarily intensive processing while
        # preserving information
        raster = self.get_raw_raster()

        if self.spatial_buffer:
            raster = self.get_spatially_buffered_raster(raster)

        return self.get_resampled_raster(raster)
        # return raster


class ProcessML:
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
                ReturnRaster(
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
        df_X, df_y = functions_creche.ds_to_ml_ready(
            xa_ds,
            predictand=self.return_predictand(),
            pos_neg_ratio=self.pos_neg_ratio,
            depth_mask_lims=self.depth_mask_lims,
        )

        trains, tests, vals = functions_creche.train_test_val_split(
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
                "train_pos_neg_ratio": float(
                    functions_creche.calc_non_zero_ratio(trains[1])
                ),
                "test_pos_neg_ratio": float(
                    functions_creche.calc_non_zero_ratio(tests[1])
                ),
                "val_pos_neg_ratio": float(
                    functions_creche.calc_non_zero_ratio(vals[1])
                ),
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


class RunML:
    def __init__(
        self,
        trains: tuple = None,
        tests: tuple = None,
        vals: tuple = None,
        config_info: dict = None,
        additional_info: dict = None,  # hacky, but used to get summary of ds into new config file
    ):
        self.trains = trains
        self.tests = tests
        self.vals = vals
        self.regressor_classification_threshold = config_info[
            "regressor_classification_threshold"
        ]
        self.n_samples = config_info["hyperparameter_search"]["n_samples"]
        self.model_code = config_info["model_code"]
        self.cv_folds = config_info["hyperparameter_search"]["cv_folds"]
        self.n_iter = config_info["hyperparameter_search"]["n_iter"]
        self.do_search = config_info["hyperparameter_search"]["do_search"]
        self.n_trials = config_info["hyperparameter_search"]["n_trials"]
        self.search_type = config_info["hyperparameter_search"]["type"]
        self.do_train = config_info["do_train"]
        self.do_save_model = config_info["do_save_model"]
        self.config_info = config_info
        self.additional_info = additional_info
        if config_info:
            self.__dict__.update(config_info)

    def initialise_model(self, params: dict = None):
        # get instance of any model type
        return functions_creche.ModelInitializer(
            model_type=self.model_code, params=params
        ).get_model()

    def threshold_datasets(self, dss: list[tuple[pd.DataFrame, pd.Series]]):
        # return list of tuples of thresholded datasets
        for ds in dss:
            yield self.threshold_dataset(ds)

    def threshold_dataset(
        self,
        ds: tuple[pd.DataFrame, pd.Series],
        # trains=None, tests=None, vals=None
    ):
        if (
            functions_creche.ModelInitializer(
                model_type=self.model_code
            ).get_data_type()
            == "discrete"
        ):
            return ds[0], functions_creche.cont_to_class(ds[1])
        else:
            return ds

    def xgboost_formatting(self, trains=None, tests=None, vals=None):
        if "xgb" in self.model_code:
            # convert data to DMatrix format
            dtrains = xgb.DMatrix(trains[0], trains[1])
            dtests = xgb.DMatrix(tests[0], tests[1])
            dvals = xgb.DMatrix(vals[0], vals[1])
            return dtrains, dtests, dvals
        else:
            return trains, tests, vals

    def get_param_search_grid(self, params: dict = None):
        # if no params provided, use default values
        if not params:
            # if param_search is True, run grid search or random search dependent on input
            if self.config_info["hyperparameter_search"]["type"] == "random":
                return functions_creche.ModelInitializer(
                    model_type=self.model_code
                ).get_random_search_grid()
            elif (
                self.config_info["hyperparameter_search"]["type"] == "grid"
            ):  # TODO: allow chaining of random and grid searches (including saving optimal values)
                return functions_creche.ModelInitializer(
                    model_type=self.model_code
                ).get_grid_search_grid()
            else:
                raise ValueError(
                    f"Parameter search type {self.config_info['hyperparameter_search']['type']} not recognised."
                )
        else:
            print("ASdfasd")

    def get_param_search(self, search_type: str, search_grid: dict = None):
        model = self.initialise_model()
        # if search grid not specified (i.e. random search?)
        if not search_grid:
            search_grid = self.get_param_search_grid()

        if search_type == "random":
            return RandomizedSearchCV(
                model,
                search_grid,
                cv=self.cv_folds,
                n_iter=self.n_iter,
                verbose=1,
                n_jobs=64,
            )
        elif search_type == "grid":
            search_grid = functions_creche.generate_gridsearch_parameter_grid(
                search_grid
            )  # currently not able to specify number of values for grid search
            return GridSearchCV(model, search_grid, cv=self.cv_folds, verbose=1)
        else:
            raise ValueError(
                f"Parameter search type {self.config_info['hyperparameter_search']['type']} not recognised."
            )

    def do_parallel_search(self, search_object):
        print(f"\nRunning parameter search for {self.model_code}...")
        with joblib.parallel_config(backend="dask", verbose=1):
            search_object.fit(
                self.trains[0][: self.n_samples],
                self.trains[1][: self.n_samples],
            )
        return search_object.best_params_

    def fetch_best_params_from_config(self):
        best_params = None
        # Try to fetch best_grid_params first, then best_random_params
        try:
            params_fp = self.config_info["file_paths"]["best_grid_params"]
            print(f"Loading best parameters from {params_fp}...")
            best_params = file_ops.read_pkl(params_fp)
        except KeyError:
            try:
                params_fp = self.config_info["file_paths"]["best_random_params"]
                print(f"Loading best parameters from {params_fp}...")
                best_params = file_ops.read_pkl(params_fp)
            except KeyError:
                print(
                    "No best parameters file(s) found in CONFIG file. Using default values instead."
                )
        return best_params

    def save_param_search(
        self, fp_root: str | Path, best_params: dict, search_type: str
    ):
        if search_type == "grid":
            params_fp = f"{fp_root}_PARAM_GRID.pickle"
        elif search_type == "random":
            params_fp = f"{fp_root}_PARAM_RANDOM.pickle"
        if not best_params:  # if best_params are None (not specified)
            params_fp = f"{fp_root}_PARAM_GENERIC.pickle"

        print(f"\nSaving best parameters to {params_fp}...")
        file_ops.write_pkl(params_fp, best_params)
        return params_fp

    def do_param_searches(self, fp_root: Path | str):
        if self.do_search:
            search_types = self.config_info["hyperparameter_search"]["search_types"]
            # if random and grid
            if len(search_types) > 2:
                print(f"Unexpected number of search types: {search_types}.")
            elif all(substring in search_types for substring in ["grid", "random"]):
                # do random
                search_object = self.get_param_search(search_type="random")
                print("RANDOM SEARCH")
                best_params = self.do_parallel_search(search_object)
                # save parameters
                self.save_param_search(fp_root, best_params, search_type="random")
                # do grid
                search_object = self.get_param_search(
                    search_type="grid", search_grid=best_params
                )
                print("GRID SEARCH")
                best_params = self.do_parallel_search(search_object)
                # save parameters
                self.save_param_search(fp_root, best_params, search_type="grid")
            # if can find random parameter file, but not grid, and grid specified run grid on random
            elif (
                Path(f"{fp_root}_PARAM_RANDOM.pickle").exists()
                and "grid" in search_types
            ):
                search_object = self.get_param_search(search_type="grid")
                best_random_params = self.fetch_best_params_from_config()
                best_params = self.do_parallel_search(
                    search_object, search_type="grid", search_grid=best_random_params
                )
                # save parameters
                self.save_param_search(fp_root, best_params, search_type="grid")
            # if only one specified, do it? ####
            else:
                search_object = self.get_param_search(search_type=search_types[0])
                best_params = self.do_parallel_search(search_object)
        else:
            # returns best of best params (grid first, then random, else None)
            best_params = self.fetch_best_params_from_config()

        return best_params

    def construct_fp_dir(self):
        res_str = utils.replace_dot_with_dash(str(self.config_info["resolution"]))
        fp_dir = Path(f"runs/{res_str}d/{self.model_code}/")
        fp_dir.mkdir(parents=True, exist_ok=True)
        return fp_dir

    def construct_fp_stem(self, fp_dir: Path | str, suffix: str = "pickle"):
        return "_".join(self.config_info["datasets"])

    def unique_identifier(self, fp_dir: Path | str, fp_stem: str):
        counter = 0
        new_filename = f"ID000_{fp_stem}_CONFIG"
        while list(fp_dir.glob(f"{new_filename}.*")) != []:
            counter += 1
            new_filename = f"ID{utils.pad_number_with_zeros(counter, resulting_len=3)}_{Path(fp_stem).stem}_CONFIG"
        return f"ID{utils.pad_number_with_zeros(counter, resulting_len=3)}"

    def train_model(self, fp_root: Path | str, hyperparams: dict = None):
        # initialise model with provided hyperparameters
        model = self.initialise_model(params=hyperparams)
        # train model
        if self.do_train:
            print(
                f"\n\nTraining the {self.model_code} model on {self.n_samples} datapoints..."
            )
            # sklearn models
            if "xgb" not in self.model_code:
                model.fit(self.trains[0], self.trains[1])
            # xgboost models
            elif self.config_info["ds_type"] == "static":
                trains, _, _ = self.xgboost_formatting(
                    self.trains, self.tests, self.vals
                )
                model = xgb.train(hyperparams, trains)
            else:
                print("timeseries models yet to be implemented")

        if self.config_info["do_save_model"]:
            model_fp = f"{fp_root}_MODEL.pickle"
            print(f"Saving model to {model_fp}...")
            file_ops.write_pkl(model_fp, model)

        return model

    def run_model(self):
        # threshold labels if necessary
        self.trains, self.tests, self.vals = self.threshold_datasets(
            [self.trains, self.tests, self.vals]
        )

        # generate path for saving data
        save_dir = self.construct_fp_dir()
        save_fp_stem = self.construct_fp_stem(save_dir)
        # generate unique ID code dependent on which files already exist
        unique_identifier = self.unique_identifier(
            fp_dir=save_dir, fp_stem=save_fp_stem
        )
        fp_root = save_dir / f"{unique_identifier}_{save_fp_stem}"

        search_start_time = time.time()
        best_params = self.do_param_searches(fp_root)
        search_time = time.time() - search_start_time
        params_fp = f"{fp_root}_PARAMS.pickle"
        file_ops.write_pkl(params_fp, best_params)

        train_start_time = time.time()
        model = self.train_model(fp_root=fp_root, hyperparams=best_params)
        train_time = time.time() - train_start_time
        # save config file
        config_fp = f"{fp_root}_CONFIG.yaml"
        file_ops.save_yaml(config_fp, self.config_info)

        # add file information to yaml
        file_info = {
            "file_paths": {
                "model": f"{fp_root}_MODEL.pickle" if self.do_save_model else None,
                "best_params": f"{fp_root}_PARAMS.pickle" if self.do_search else None,
                "config": f"{fp_root}_CONFIG.yaml",
                "search_time": f"{search_time:.02f}s",
                "train_time": f"{train_time:.02f}s",
            },
            "additional_info": self.additional_info if self.additional_info else None,
        }
        file_ops.edit_yaml(config_fp, file_info)
        return model


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
        data_type = functions_creche.ModelInitializer(
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
                functions_creche.cont_to_class(
                    y, self.config_info["regressor_classification_threshold"]
                ),
                functions_creche.cont_to_class(
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
        run_dir = Path(self.config_info["file_paths"]["config"]).parent
        plot_dir_id = self.config_info["file_paths"]["config"].split("_CONFIG")[0]
        plot_dir_child = (
            f"{plot_dir_id}/{self.ds_type}" if self.ds_type else plot_dir_id
        )
        plot_dir = run_dir / "plots" / plot_dir_child
        plot_dir.mkdir(parents=True, exist_ok=True)

        return plot_dir

    def save_fig(self, fn: str):
        if self.save_graphs:
            plot_dir = self.get_plot_dir()
            plt.savefig(plot_dir / f"{fn}.png")

    def plot_confusion_matrix(self, y, predictions):
        model_results.plot_confusion_matrix(
            labels=y,
            predictions=predictions,
            label_threshold=self.config_info["regressor_classification_threshold"],
            presentation_format=self.presentation_format,
        )
        self.save_fig(fn="confusion_matrix")

    def plot_regression(self, y, predictions):
        model_results.plot_regression_histograms(
            y, predictions, presentation_format=self.presentation_format
        )
        self.save_fig(fn="regression")

    def plot_spatial_inference_comparison(self, y, predictions):
        model_results.plot_spatial_inference_comparison(
            y, predictions, self.presentation_format
        )
        self.save_fig(fn="spatial_inference_comparison")

    def plot_spatial_confusion_matrix(self, y, predictions):

        confusion_values = model_results.plot_spatial_confusion_matrix(
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
        data_type = functions_creche.ModelInitializer(
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


def run_config_files(fps: list[str]):
    for fp in tqdm(fps, desc="Running models according to config file(s)"):
        config_info = file_ops.read_yaml(fp)
        # get data
        (trains, tests, vals), ds_info = ProcessML(
            config_info=config_info
        ).generate_ml_ready_data()
        # train model
        model = RunML(
            trains=trains,
            tests=tests,
            vals=vals,
            config_info=config_info,
            additional_info=ds_info,
        ).run_model()
        # analyse results
        AnalyseResults(
            model=model,
            trains=trains,
            tests=tests,
            vals=vals,
            config_info=config_info,
        ).analyse_results()


# def preprocess_data(df, label_column):
#     # Split dataframe into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(
#         df.loc[:, df.columns != label_column],
#         df[label_column],
#         test_size=0.2,
#         random_state=42,
#     )

#     # Min-max scaling ignoring NaNs
#     scaler = MinMaxScaler()
#     X_train_scaled = scaler.fit_transform(X_train.drop(columns=[label_column]))
#     X_test_scaled = scaler.transform(X_test.drop(columns=[label_column]))

#     # Convert scaled arrays back to Dask DataFrame
#     columns = X_train.drop(columns=[label_column]).columns
#     X_train_scaled_df = dd.from_dask_array(
#         pd.DataFrame(X_train_scaled, columns=columns)
#     )
#     X_test_scaled_df = dd.from_dask_array(pd.DataFrame(X_test_scaled, columns=columns))

#     # Generate a column showing 1 if any NaN is in the row, 0 otherwise
#     X_train_scaled_df["onehot_nan"] = X_train_scaled_df.isnull().any(axis=1).astype(int)
#     X_test_scaled_df["onehot_nan"] = X_test_scaled_df.isnull().any(axis=1).astype(int)

#     # Replace NaNs with 0
#     X_train_scaled_df = X_train_scaled_df.fillna(0)
#     X_test_scaled_df = X_test_scaled_df.fillna(0)

#     return X_train_scaled_df, X_test_scaled_df

# I THINK THIS HAS BEEN REPLACED BY THE DATA CLASS DEFINTION BUT IT'S HERE IN CASE
# def generate_gebco_xarray(
#     resolution: float = 0.1,
#     lats: list[float, float] = [-40, 0],
#     lons: list[float, float] = [130, 170],
# ):
#     # TODO: make generic for other bathymetry data?

#     # check if gebco xarray already exists
#     # res_str = utils.replace_dot_with_dash(str(resolution))

#     gebco_xa_dir = Path(config.bathymetry_folder) / "gebco/rasters"
#     gebco_xa_dir.mkdir(parents=True, exist_ok=True)

#     # check if there exists a file spanning an adequate spatial extent
#     nc_fps = list(Path(gebco_xa_dir).rglob("*.nc"))
#     subset_fps = functions_creche.find_files_for_area(nc_fps, lats, lons)
#     if len(subset_fps) > 0:
#         # arbitrarily select first suitable file
#         gebco_fp = subset_fps[0]
#     else:
#         raise FileNotFoundError("No GEBCO files with suitable geographic range found.")
#     # TODO: proper chunking
#     return xa.open_dataset(gebco_fp)

# spatial_extent_info = functions_creche.lat_lon_string_from_tuples(lats, lons)
# gebco_xa_fp = gebco_xa_dir / f"gebco_{res_str}_{spatial_extent_info}.nc"

# # if not, generate
# spatial_extent_info = functions_creche.lat_lon_string_from_tuples(lats, lons)
# gebco_xa_fp = gebco_xa_dir / f"gebco_{res_str}_{spatial_extent_info}.nc"

# if gebco_xa_fp.exists():
#     print(f"Gebco xarray at {res_str} already exists.")
#     return xa.open_dataset(gebco_xa_fp)
# else:
#     # look for appropriate file which spans range
#     nc_fps = list(Path(gebco_xa_dir).rglob("*.nc"))

#     # TODO: incorporate summation of multiple files
#     gebco_xa = spatial_data.process_xa_d(xa.open_dataset(gebco_fp, chunks="auto"))

#     # determine spatial extent of GEBCO file
#     lats = functions_creche.get_min_max_coords(gebco_xa, "latitude")
#     lons = functions_creche.get_min_max_coords(gebco_xa, "longitude")
#     # update spatial extent info
#     spatial_extent_info = functions_creche.lat_lon_string_from_tuples(lats, lons)
#     gebco_xa_fp = gebco_xa_dir / f"gebco_{res_str}_{spatial_extent_info}.nc"

#     print(f"generating GEBCO raster at {res_str} degree resolution...")
#     # resample if necessary
#     gebco_xa = functions_creche.resample_xa_d(gebco_xa, resolution=resolution)

#     # save to filepath
#     print(f"saving GEBCO raster to {gebco_xa_fp}...")
#     gebco_xa.to_netcdf(gebco_xa_fp)

#     return gebco_xa

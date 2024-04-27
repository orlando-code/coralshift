# general
import numpy as np
import pandas as pd

# spatial
import xarray as xa
import dask_geopandas as daskgpd
import shapely.geometry as sgeometry
import geopandas as gpd
from rasterio import features as featuresio
import rasterio
import xesmf as xe
from rasterio.enums import MergeAlg
from pyinterp.backends import xarray
from pyinterp import fill


# file ops
from tqdm.auto import tqdm
from pathlib import Path

# custom
from coralshift.dataloading import bathymetry
from coralshift.utils import utils, config
from coralshift.processing import spatial_data, ml_processing

import coralshift.cmipper.file_ops as cmipper_file_ops

# from coralshift.dataloading.cmipper.file_ops import find_intersecting_cmip


""" Ground truths """


def generate_unep_xarray(
    lats: list[float, float] = [-90, 90],
    lons: list[float, float] = [-180, 180],
    degrees_resolution: float = 15 / 3600,
):
    # check if unep xarray already exists
    res_str = utils.replace_dot_with_dash(str(round(degrees_resolution, 3)))

    unep_xa_dir = Path(config.gt_data_dir) / "unep_wcmc/rasters"
    unep_xa_dir.mkdir(parents=True, exist_ok=True)

    spatial_extent_info = utils.lat_lon_string_from_tuples(lats, lons).upper()
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
        unep_raster = rasterize_geodf(filtered_gdf, resolution=degrees_resolution)

        print("casting raster to xarray...")
        # generate gt xarray
        unep_xa = raster_to_xarray(
            unep_raster,
            x_y_limits=utils.lat_lon_vals_from_geo_df(filtered_gdf)[:4],
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
        lambda x: [utils.try_convert_to_float(val.strip()) for val in x]
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


""" Bathymetry/seafloor slopes """


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
            return bathymetry.generate_gebco_xarray(self.lats, self.lons)
        elif self.dataset in ["gebco_slope", "bathymetry_slope"]:
            return bathymetry.generate_gebco_slopes_xarray(self.lats, self.lons)
        elif self.dataset in ["cmip6", "cmip"]:
            # ensure necessary files downloaded: variables, years, lats, lons, levs
            # TODO: (probably) – split up download and processing, ensuring download for all variables
            # finishes before processing. May also involve changing how variables are passed (i.e. call
            # to processing will take a list of variables and will only be performed once)
            # running into issues with wider area due to [0,-50] tos file apparently being text????
            # ensure_cmip6_downloaded(
            #     variables=self.env_vars,
            #     lats=self.lats,
            #     lons=self.lons,
            #     levs=self.levs,
            # )
            # TODO: potentially tidy variable assignment in find_intersecting_cmip
            # TODO: this returns all variables in the file searched, not the subset. Also returns tim_bnds as a variable
            raster, _ = cmipper_file_ops.find_intersecting_cmip(
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

        current_resolution = utils.get_resolution(raster)
        if self.resolution < current_resolution:
            resample_method = self.upsample_method
        else:
            resample_method = self.downsample_method

        # doing this to get around "buffer size too small error"
        rough_regrid = resample_xa_d(
            raster,
            lat_range=self.lats,
            lon_range=self.lons,
            resolution=self.resolution,
            resample_method=resample_method,
        )
        return xesmf_regrid(
            rough_regrid,
            lat_range=self.lats,
            lon_range=self.lons,
            resolution=self.resolution,
            # resample_method=resample_method,
        )

    def get_spatially_buffered_raster(self, raster):
        print("\tapplying spatial buffering...")
        return apply_fill_loess(raster, nx=self.spatial_buffer, ny=self.spatial_buffer)

    def process_timeseries_to_static(self, raster):
        # TODO: add processing for other timseries datasets
        if self.dataset in ["cmip6", "cmip"]:
            print("\tcalculating statistics for static ML model(s)...")
            static_ds = ml_processing.calculate_statistics(
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
        processed_raster = spatial_data.process_xa_d(
            self.get_raw_raster()
        )  # this shouldn't be necessary (process_xa_d included in cmip download, but made a change to remove time_bnds
        # since then)

        if self.spatial_buffer:
            raster = self.get_spatially_buffered_raster(processed_raster)

        return self.get_resampled_raster(raster)
        # return raster


def rasterize_geodf(
    geo_df: gpd.geodataframe,
    resolution: float = 1.0,
    all_touched: bool = True,
) -> np.ndarray:
    """Rasterize a geodataframe to a numpy array.

    Args:
        geo_df (gpd.geodataframe): Geodataframe to rasterize
        resolution (float): Resolution of the raster in degrees

    Returns:
        np.ndarray: Rasterized numpy array
    TODO: add crs customisation. Probably from class object elsewhere.
    Currently assumes EPSG:4326.
    """

    xmin, ymin, xmax, ymax, width, height = utils.lat_lon_vals_from_geo_df(
        geo_df, resolution
    )
    # Create the transform based on the extent and resolution
    transform = rasterio.transform.from_bounds(xmin, ymin, xmax, ymax, width, height)
    transform.crs = rasterio.crs.CRS.from_epsg(4326)

    # Any chance of a loading bar? No: would have to dig into the function istelf.
    # could be interesting...
    return featuresio.rasterize(
        [(shape, 1) for shape in geo_df["geometry"]],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        all_touched=all_touched,
        dtype=rasterio.uint8,
        merge_alg=MergeAlg.add,
    )


def raster_to_xarray(
    raster: np.ndarray,
    x_y_limits: np.ndarray,
    resolution: float = 1.0,
    name: str = "raster_cast_to_xarray",
) -> xa.DataArray:
    """Convert a raster to an xarray DataArray.

    Args:
        raster (np.ndarray): Raster to convert
        resolution (float): Resolution of the raster in degrees

    Returns:
        xa.DataArray: DataArray of the raster
    TODO: add attributes kwarg
    """

    lon_min, lat_min, lon_max, lat_max = x_y_limits
    cell_width = int((lon_max - lon_min) / resolution)
    cell_height = int((lat_max - lat_min) / resolution)

    # Create longitude and latitude arrays
    longitudes = np.linspace(lon_min, lon_max, cell_width)
    # reversed because raster inverted
    latitudes = np.linspace(lat_max, lat_min, cell_height)

    # Create an xarray DataArray with longitude and latitude coordinates
    xa_array = xa.DataArray(
        raster,
        coords={"latitude": latitudes, "longitude": longitudes},
        dims=["latitude", "longitude"],
        name=name,
    )
    # Set the CRS (coordinate reference system) if needed
    # TODO: make kwarg
    xa_array.attrs["crs"] = "EPSG:4326"  # Example CRS, use the appropriate CRS
    # TODO: set attributes if required
    #     attrs=dict(
    #         description="Rasterised Reef Check coral presence survey data"
    #     ))
    return spatial_data.process_xa_d(xa_array)


def rasterise_points_df(
    df: pd.DataFrame,
    lat_column: str,
    lon_column: str,
    resolution: float = 1.0,
    bbox: list[float] = [-90, -180, 90, 180],
) -> np.ndarray:
    """Rasterize a pandas dataframe of points to a numpy array.

    Args:
        df (pd.DataFrame): Dataframe of points to rasterize
        resolution (float): Resolution of the raster in degrees

    Returns:
        np.ndarray: Rasterized numpy array"""

    # extract bbox limits of your raster
    min_lat, min_lon, max_lat, max_lon = bbox

    # Calculate the number of rows and columns in the raster
    num_rows = int((max_lat - min_lat) / resolution)
    num_cols = int((max_lon - min_lon) / resolution)

    # Initialize an empty raster (of zeros)
    raster = np.zeros((num_rows, num_cols), dtype=int)

    # Convert latitude and longitude points to row and column indices of raster
    row_indices = ((max_lat - df[lat_column]) // resolution).astype(int)
    col_indices = ((df[lon_column] - min_lon) // resolution).astype(int)

    # Filter coordinates that fall within the bounding box: this produces a binary mask
    valid_indices = (
        (min_lat <= df[lat_column])
        & (df[lat_column] <= max_lat)
        & (min_lon <= df[lon_column])
        & (df[lon_column] <= max_lon)
    )

    # # Update the raster with counts of valid coordinates
    raster[row_indices[valid_indices], col_indices[valid_indices]] += 1

    # list of row, column indices corresponding to each latitude/longitude point
    valid_coordinates = list(
        zip(row_indices[valid_indices], col_indices[valid_indices])
    )
    # count number of repeated index pairs and return unique
    unique_coordinates, counts = np.unique(
        valid_coordinates, axis=0, return_counts=True
    )
    # assign number of counts to each unique raster
    raster[unique_coordinates[:, 0], unique_coordinates[:, 1]] = counts

    return raster


def apply_fill_loess(dataset: xa.Dataset, nx=2, ny=2):
    """
    Apply fill.loess to each time step for each variable in the xarray dataset.

    Args:
        dataset (xarray.Dataset): Input xarray dataset with time series of variables.
        nx (int): Number of pixels to extend in the x-direction.
        ny (int): Number of pixels to extend in the y-direction.

    Returns:
        xarray.Dataset: Buffered xarray dataset.
    """
    # TODO: nested tqdm in notebooks and scripts
    # Create a copy of the original dataset
    buffered_dataset = dataset.copy(deep=True)
    buffered_data_vars = buffered_dataset.data_vars

    print(f"{len(buffered_data_vars)} raster(s) to spatially buffer...")
    for _, (var_name, var_data) in tqdm(
        enumerate(buffered_data_vars.items()),
        desc="Buffering variables",
        total=len(buffered_data_vars),
        position=0,
    ):  # for each variable in the dataset
        if utils.check_var_has_coords(
            var_data
        ):  # if dataset has latitude, longitude, and time coordinates
            for t in tqdm(
                buffered_dataset.time,
                desc=f"Processing timesteps of variable '{var_name}'",
                leave=False,
                position=1,
            ):  # buffer each timestep
                grid = xarray.Grid2D(var_data.sel(time=t))
                filled = fill.loess(grid, nx=nx, ny=ny)
                buffered_data_vars[var_name].loc[dict(time=t)] = filled.T
        elif utils.check_var_has_coords(
            var_data, ["latitude", "longitude"]
        ):  # if dataset has latitude, longitude only
            # for var_name in tqdm(list(buffered_dataset.data_vars.keys()), desc="Processing variables..."):  #
            grid = xarray.Grid2D(var_data, geodetic=False)
            filled = fill.loess(grid, nx=nx, ny=ny)
            # buffered_data_vars[var_name].loc[dict(var=var)] = filled.T
            buffered_dataset.update(
                {var_name: (buffered_dataset[var_name].dims, filled)}
            )
        else:
            print(
                f"""Variable must have at least 'latitude', and 'longitude' coordinates to be spatially padded.
                \nVariable {var_name} has '{var_data.coords}'. Skipping..."""
            )

    return buffered_dataset


def resample_xa_d(
    xa_d: xa.DataArray | xa.Dataset,
    lat_range: list[float] = None,
    lon_range: list[float] = None,
    resolution: float = 0.1,
    resample_method: str = "linear",
):
    """
    Resample an xarray DataArray or Dataset to a common extent and resolution.

    Args:
        xa_d (xa.DataArray | xa.Dataset): xarray DataArray or Dataset to resample.
        lat_range (list[float]): Latitude range of the common extent.
        lon_range (list[float]): Longitude range of the common extent.
        resolution (float): Longitude resolution of the common extent.
        resample_method (str, optional): Resampling method to use. Defaults to "linear".

    Returns:
        xa.DataArray | xa.Dataset: Resampled xarray DataArray or Dataset.

    resample_methods:
        "linear" – Bilinear interpolation.
        "nearest" – Nearest-neighbor interpolation.
        "zero" – Piecewise-constant interpolation.
        "slinear" – Spline interpolation of order 1.
        "quadratic" – Spline interpolation of order 2.
        "cubic" – Spline interpolation of order 3.
        TODO: implement"polynomial"
    """
    # if coordinate ranges not specified, infer from present xa_d
    if not (lat_range and lon_range):
        lat_range = spatial_data.min_max_of_coords(xa_d, "latitude")
        lon_range = spatial_data.min_max_of_coords(xa_d, "longitude")

    lat_range = sorted(lat_range)
    lon_range = sorted(lon_range)

    # Create a dummy dataset with the common extent and resolution
    common_dataset = xa.Dataset(
        coords={
            "latitude": (
                ["latitude"],
                np.arange(lat_range[0], lat_range[1] + resolution, resolution),
            ),
            "longitude": (
                ["longitude"],
                np.arange(lon_range[0], lon_range[1] + resolution, resolution),
            ),
        }
    )
    current_resolution = utils.get_resolution(xa_d)
    # if upsampling, interpolate
    if resolution < current_resolution:
        return xa_d.sel(
            latitude=slice(*lat_range), longitude=slice(*lon_range)
        ).interp_like(common_dataset, method=resample_method)
    # if downsampling, coarsen
    else:
        return coarsen_xa_d(
            xa_d.sel(latitude=slice(*lat_range), longitude=slice(*lon_range)),
            resolution,
            resample_method,
        )


# @lru_cache(maxsize=None)  # Set maxsize to limit the cache size, or None for unlimited
def xesmf_regrid(
    xa_d: xa.DataArray | xa.Dataset,
    lat_range: list[float] = None,
    lon_range: list[float] = None,
    resolution: float = 0.1,
    resampled_method: str = "bilinear",
):

    lon_range = sorted(lon_range)
    lat_range = sorted(lat_range)
    target_grid = xe.util.grid_2d(
        lon_range[0],
        lon_range[1],
        resolution,
        lat_range[0],
        lat_range[1],
        resolution,  # longitude range and resolution
    )  # latitude range and resolution

    regridder = xe.Regridder(xa_d, target_grid, method=resampled_method)

    # return spatial_data.process_xa_d(regridder(xa_d))
    return process_xesmf_regridded(regridder(xa_d))


def process_xesmf_regridded(
    xa_d: xa.DataArray | xa.Dataset,
):
    xa_d["lon"] = xa_d.lon.values[0, :]
    xa_d["lat"] = xa_d.lat.values[:, 0]

    return xa_d.rename(
        {"x": "longitude", "y": "latitude", "lon": "longitude", "lat": "latitude"}
    )


def coarsen_xa_d(xa_d, resolution: float = 0.1, method="mean"):
    # TODO: for now, treating lat and long with indifference (since this is how data is).
    num_points_lat = int(
        round(resolution / abs(xa_d["latitude"].diff("latitude").mean().values))
    )
    num_points_lon = int(
        round(resolution / abs(xa_d["longitude"].diff("longitude").mean().values))
    )

    return xa_d.coarsen(
        latitude=num_points_lat,
        longitude=num_points_lon,
        boundary="pad",
    ).reduce(method)


def resample_to_other(
    xa_d_to_resample, target_xa, resample_method: str = "linear"
) -> xa.Dataset | xa.DataArray:
    """
    Resample an xarray DataArray or Dataset to the resolution and extent of another xarray DataArray or Dataset.

    Args:
        xa_d_to_resample (xa.DataArray | xa.Dataset): xarray DataArray or Dataset to resample.
        target_xa (xa.DataArray | xa.Dataset): xarray DataArray or Dataset to resample to.

    Returns:
        xa.DataArray | xa.Dataset: Resampled xarray DataArray or Dataset.

    resample_methods:
        "linear" – Bilinear interpolation.
        "nearest" – Nearest-neighbor interpolation.
        "zero" – Piecewise-constant interpolation.
        "slinear" – Spline interpolation of order 1.
        "quadratic" – Spline interpolation of order 2.
        "cubic" – Spline interpolation of order 3.
        TODO: implement"polynomial"
    """
    return xa_d_to_resample.interp_like(target_xa, method=resample_method)


def spatially_combine_xa_d_list(
    xa_d_list: list[xa.DataArray | xa.Dataset],
    lat_range: list[float],
    lon_range: list[float],
    resolution: float,
    resample_method: str = "linear",
) -> xa.Dataset:
    """
    Resample and merge a list of xarray DataArrays or Datasets to a common extent and resolution.

    Args:
        xa_d_list (list[xa.DataArray | xa.Dataset]): List of xarray DataArrays or Datasets to resample and merge.
        lat_range (list[float]): Latitude range of the common extent.
        lon_range (list[float]): Longitude range of the common extent.
        resolution (float): Resolution of the common extent.
        resample_method (str, optional): Resampling method to use. Defaults to "linear".

    Returns:
        xa.Dataset: Dataset containing the resampled and merged DataArrays or Datasets.

    N.B. resample_method can take the following values for 1D interpolation:
        - "linear": Linear interpolation.
        - "nearest": Nearest-neighbor interpolation.
        - "zero": Piecewise-constant interpolation.
        - "slinear": Spline interpolation of order 1.
        - "quadratic": Spline interpolation of order 2.
        - "cubic": Spline interpolation of order 3.
    and these for n-d interpolation:
        - "linear": Linear interpolation.
        - "nearest": Nearest-neighbor interpolation.
    """

    # Create a new dataset with the common extent and resolution
    common_dataset = xa.Dataset(
        coords={
            "latitude": (["latitude"], np.arange(*np.sort(lat_range), resolution)),
            "longitude": (
                ["longitude"],
                np.arange(*np.sort(lon_range), resolution),
            ),
        }
    )

    # Iterate through input datasets, resample, and merge into the common dataset
    for input_ds in tqdm(
        xa_d_list, desc=f"resampling and merging {len(xa_d_list)} datasets"
    ):
        # Resample the input dataset to the common resolution and extent using bilinear interpolation
        resampled_dataset = input_ds.interp(
            latitude=common_dataset["latitude"].sel(
                latitude=slice(min(lat_range), max(lat_range))
            ),
            longitude=common_dataset["longitude"].sel(
                longitude=slice(min(lon_range), max(lon_range))
            ),
            method=resample_method,
        )

        # Merge the resampled dataset into the common dataset
        common_dataset = xa.merge(
            [common_dataset, resampled_dataset], compat="no_conflicts"
        )

    return common_dataset


def adaptive_depth_mask(
    df,
    depth_mask_lims=[-50, 0],
    pos_neg_ratio=0.1,
    tolerance=0.005,
    remove_rows=True,
    predictand="UNEP_GDCR",
    depth_var="elevation",
):
    hold_df = df.copy()
    depth_mask_lims = sorted(depth_mask_lims)
    # where value in df is not zero
    non_zero_ratio = utils.calc_non_zero_ratio(df, predictand)
    # print(non_zero_ratio)

    best_non_zero_ratio_diff = 1
    # Iterate for a fixed number of times
    for i in range(1000):
        prev_df = df.copy()  # Store previous depth_mask_lims

        if (
            non_zero_ratio >= pos_neg_ratio - tolerance
            and non_zero_ratio <= pos_neg_ratio + tolerance
        ):
            return df  # Exit the loop if pos_neg_ratio is within tolerance

        # TODO: adjust depending on depth_mask_lims_size?
        if non_zero_ratio > pos_neg_ratio + tolerance:
            # increase minimum depth
            depth_mask_lims[0] -= 1
        elif non_zero_ratio < pos_neg_ratio - tolerance:
            # decrease minimum depth
            depth_mask_lims[0] += 1

        df = depth_filter(df, depth_mask_lims, remove_rows, depth_var)
        non_zero_ratio = utils.calc_non_zero_ratio(df, predictand)

        # Check for nan non_zero_ratio
        if np.isnan(non_zero_ratio):
            df = prev_df  # Restore previous depth_mask_lims
            break

        pos_neg_non_zero_diff = abs(pos_neg_ratio - non_zero_ratio)
        if pos_neg_non_zero_diff < best_non_zero_ratio_diff:
            best_non_zero_ratio_diff = pos_neg_non_zero_diff
            best_ratio_depth_mask_lims = depth_mask_lims.copy()

        print(f"{i} depth mask", depth_mask_lims)
        print("pos_neg_ratio", non_zero_ratio)

    print(best_ratio_depth_mask_lims)
    # if loop not satisfied, return depth_mask_lims which get closest to pos_neg_ratio
    return depth_filter(hold_df, best_ratio_depth_mask_lims, remove_rows, depth_var)


def depth_filter(
    df: pd.DataFrame,
    depth_mask_lims: list[float, float],
    remove_rows: bool = True,
    depth_var: str = "elevation",
):
    df_depth = df.copy()
    # generate boolean depth mask
    depth_condition = (df_depth[depth_var] < max(depth_mask_lims)) & (
        df_depth[depth_var] > min(depth_mask_lims)
    )
    # if remove_rows (default), remove rows outside of depth mask
    if remove_rows:
        df_depth = df_depth[depth_condition]
    # if not remove_rows, add a column of 1s and 0s to indicate whether row is within depth mask
    else:
        df_depth["within_depth"] = 0
        df_depth.loc[depth_condition, "within_depth"] = 1

    return df_depth

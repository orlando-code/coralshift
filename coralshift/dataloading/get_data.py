# general
import numpy as np
import pandas as pd
import re

# spatial
import xarray as xa
import dask_geopandas as daskgpd
import shapely.geometry as sgeometry
import geopandas as gpd
from rasterio import features as featuresio
import rasterio
from rasterio.enums import Resampling
import xesmf as xe
from rasterio.enums import MergeAlg
from pyinterp.backends import xarray as pyxarray

# from pyinterp.core import fill
from pyinterp.fill import loess

# file ops
from tqdm.auto import tqdm
from pathlib import Path

# import functools

# custom
from coralshift.dataloading import bathymetry
from coralshift.utils import utils, config
from coralshift.utils.config import Config
from coralshift.processing import spatial_data, ml_processing

import cmipper.file_ops as cmipper_file_ops
import cmipper.utils as cmipper_utils


""" Ground truths """


def find_files_for_resolution(fps, resolution=str, start_str="unep", end_str=".nc"):
    result = []
    resolution = f"_{resolution}_"
    pattern = re.compile(rf"{start_str}{resolution}.*{end_str}", re.IGNORECASE)
    # print(pattern)
    for fp in fps:
        if pattern.search(Path(fp).name):
            result.append(fp)
    return result


def generate_xa_ds_from_shapefile(
    shapefile_id: str,
    shapefile_fp: str | Path,
    lats: list[float, float] = [-90, 90],
    lons: list[float, float] = [-180, 180],
    degrees_resolution: float = 15 / 3600,  # 15 arcseconds (gebco resolution)
):
    # check if unep xarray already exists
    res_str = utils.replace_dot_with_dash(str(round(degrees_resolution, 3)))

    xa_dir = Path(config.gt_data_dir) / f"{shapefile_id.upper()}/rasters"
    xa_dir.mkdir(parents=True, exist_ok=True)
    potential_fps = list(xa_dir.glob("*.nc"))

    # correct resolution
    correct_resolutions = find_files_for_resolution(potential_fps, res_str)
    xas = cmipper_file_ops.find_files_for_area(correct_resolutions, lats, lons)

    if len(xas) >= 10000:   # TODO: temp
        print(f"Loading {shapefile_id} xarray at {degrees_resolution:.03f} degrees resolution.")
        print(f"loading from {xas[0]}")
        return xa.open_dataset(xas[0]).sel(latitude=slice(*lats), longitude=slice(*lons))
    else:
        print(f"Loading {shapefile_id} data from original shapefile: {shapefile_fp}")

        # load unep tabular data. Don't dask yet to allow filtering by region (if required)
        gdf = daskgpd.read_file(shapefile_fp, npartitions=4)
        if shapefile_id in ["wri", "WRI_REEF_EXTENT"]:  # this gdf is weirdly bundled into one row
            gdf = gdf.explode()

        geometry_filter = sgeometry.box(min(lons), min(lats), max(lons), max(lats))
        filtered_gdf = gdf[gdf.geometry.intersects(geometry_filter)]  # coarse filter (allowing multipolys)
        filtered_gdf = filtered_gdf.compute().explode(index_parts=True)
        geometry_filter = sgeometry.box(min(lons), min(lats), max(lons), max(lats))
        filtered_gdf = filtered_gdf[filtered_gdf.geometry.intersects(geometry_filter)]  # fine filter (all polys)

        print(
            f"generating UNEP raster at {degrees_resolution:.03f} degrees resolution..."
        )
        # generate gt raster
        # Purist: defined here as the mean (lat/lon) value maximum resolution (30m) the UNEP data at the equator
        # degrees_resolution = spatial_data.distance_to_degrees(
        #     distance_lat=452, approx_lat=0, approx_lon=0
        # )[-1] (constant-area)
        raster = rasterize_geodf(filtered_gdf, resolution=degrees_resolution)

        print("casting raster to xarray...")
        # generate gt xarray
        xa_d = raster_to_xarray(
            raster,
            x_y_limits=utils.lat_lon_vals_from_geo_df(filtered_gdf)[:4],
            resolution=degrees_resolution,
            name=shapefile_id.upper(),
        ).chunk("auto")
        xa_d.longitude.attrs["units"] = "degrees_east" 
        xa_d.latitude.attrs["units"] = "degrees_north" 
        
        # reproject to bathymetry resolution for saving


        # generate filepath and save
        spatial_extent_info = cmipper_utils.lat_lon_string_from_tuples(lats, lons).upper()
        xa_d_fp = xa_dir / f"unep_{res_str}_{spatial_extent_info}.nc"
        print(f"saving UNEP raster to {xa_d_fp}...")
        xa_d.to_netcdf(xa_d_fp)

        return xa_d.to_dataset()


# def generate_unep_xarray(
#     lats: list[float, float] = [-90, 90],
#     lons: list[float, float] = [-180, 180],
#     degrees_resolution: float = 15 / 3600,  # 15 arcseconds (gebco resolution)
# ):
#     # check if unep xarray already exists
#     res_str = utils.replace_dot_with_dash(str(round(degrees_resolution, 3)))

#     unep_xa_dir = Path(config.gt_data_dir) / "unep_wcmc/rasters"
#     unep_xa_dir.mkdir(parents=True, exist_ok=True)
#     unep_fps = list(unep_xa_dir.glob("*.nc"))

#     # correct resolution
#     correct_resolutions = find_files_for_resolution(unep_fps, res_str)
#     unep_xas = cmipper_file_ops.find_files_for_area(correct_resolutions, lats, lons)

#     if len(unep_xas) >= 1:
#         print(f"Loading UNEP xarray at {degrees_resolution:.03f} degrees resolution.")
#         return xa.open_dataset(unep_xas[0]).sel(latitude=slice(*lats), longitude=slice(*lons))
#     else:
#         print("loading UNEP data...")
#         unep_fp = (
#             Path(config.gt_data_dir)
#             / "unep_wcmc/01_Data/WCMC008_CoralReef2021_Py_v4_1.shp"
#         )
#         # load unep tabular data. Don't dask yet to allow filtering by region (if required)
#         # unep_gdf = gpd.read_file(unep_fp).cx[lats[0] : lats[1], lons[0] : lons[1]]
#         unep_gdf = daskgpd.read_file(unep_fp, npartitions=4)
#         geometry_filter = sgeometry.box(min(lons), min(lats), max(lons), max(lats))
#         filtered_gdf = unep_gdf[unep_gdf.geometry.intersects(geometry_filter)]  # coarse filter (allowing multipolys)
#         filtered_gdf = filtered_gdf.compute().explode(index_parts=True)
#         geometry_filter = sgeometry.box(min(lons), min(lats), max(lons), max(lats))
#         filtered_gdf = filtered_gdf[filtered_gdf.geometry.intersects(geometry_filter)]  # fine filter (all polys)

#         print(
#             f"generating UNEP raster at {degrees_resolution:.03f} degrees resolution..."
#         )
#         # generate gt raster
#         # Purist: defined here as the mean (lat/lon) value maximum resolution (30m) the UNEP data at the equator
#         # degrees_resolution = spatial_data.distance_to_degrees(
#         #     distance_lat=452, approx_lat=0, approx_lon=0
#         # )[-1] (constant-area)
#         unep_raster = rasterize_geodf(filtered_gdf, resolution=degrees_resolution)

#         print("casting raster to xarray...")
#         # generate gt xarray
#         unep_xa = raster_to_xarray(
#             unep_raster,
#             x_y_limits=utils.lat_lon_vals_from_geo_df(filtered_gdf)[:4],
#             resolution=degrees_resolution,
#             name="UNEP_GDCR",
#         ).chunk("auto")

#         # generate filepath and save
#         spatial_extent_info = cmipper_utils.lat_lon_string_from_tuples(lats, lons).upper()
#         unep_xa_fp = unep_xa_dir / f"unep_{res_str}_{spatial_extent_info}.nc"
#         print(f"saving UNEP raster to {unep_xa_fp}...")
#         unep_xa.to_netcdf(unep_xa_fp)

#         return unep_xa.to_dataset()


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
        config_info: Config,
        # dataset: str = None,
        lats: list[float, float] = None,
        lons: list[float, float] = None,
        levs: list[int, int] = None,
        resolution: float = None,
        resolution_unit: str = None,
        # pos_neg_ratio: float = 0.1,
        upsample_method: str = None,
        downsample_method: str = None,
        spatial_buffer: int = None,
        ds_type: str = None,
        env_vars: list[str] = None,
        year_range_to_include: list[int, int] = None,
        # source: str = "EC-Earth3P-HR",    # TODO: be more specific with which source
        # member: str = "r1i1p1f1",
        # config_info: dict = None,
    ):
        # if config_info:
        #     self.__dict__.update(config_info)

        # self.dataset = dataset if dataset else config_info["dataset"]
        self.lats = lats if lats else config_info.lats
        self.lons = lons if lons else config_info.lons
        self.buffered_lats = utils.get_buffered_lims(self.lats)
        self.buffered_lons = utils.get_buffered_lims(self.lons)
        self.levs = levs if levs else config_info.levs
        self.resolution = resolution if resolution else config_info.resolution
        self.resolution_unit = (
            resolution_unit if resolution_unit else config_info.resolution_unit
        )
        # self.pos_neg_ratio = pos_neg_ratio if pos_neg_ratio else config_info["pos_neg_ratio"]
        self.upsample_method = (
            upsample_method if upsample_method else config_info.upsample_method
        )
        self.downsample_method = (
            downsample_method if downsample_method else config_info.downsample_method
        )
        self.spatial_buffer = (
            spatial_buffer if spatial_buffer else config_info.spatial_buffer
        )
        self.ds_type = ds_type if ds_type else config_info.ds_type
        self.env_vars = (
            env_vars if env_vars else config_info.env_vars if config_info else None
        )
        self.year_range_to_include = (
            year_range_to_include
            if year_range_to_include
            else config_info.year_range_to_include
        )
        self.cfg = config_info

    def get_raw_raster(self, dataset, ds=None):
        if dataset in ["unep", "unep_wcmc", "gdcr", "unep_coral_presence"]:
            # TODO: check that there isn't an intersecting one already
            return generate_xa_ds_from_shapefile(
                shapefile_id="UNEP_GDCR", shapefile_fp=config.gdcr_dir / "01_Data/WCMC008_CoralReef2021_Py_v4_1.shp",
                lats=self.buffered_lats, lons=self.buffered_lons, degrees_resolution=15/3600)   # 15" (gebco resolution)
        if dataset in ["wri"]:
            return generate_xa_ds_from_shapefile(
                shapefile_id="WRI_REEF_EXTENT", shapefile_fp=config.wri_dir / "Reefs/reef_500_poly.shp",
                lats=self.buffered_lats, lons=self.buffered_lons, degrees_resolution=15/3600)   # 15" (gebco resolution)
        elif dataset in ["gebco", "bathymetry"]:
            return bathymetry.generate_gebco_xarray(self.buffered_lats, self.buffered_lons)
        elif dataset in ["gebco_slope", "bathymetry_slope", "slope"]:
            return bathymetry.generate_gebco_slopes_xarray(self.buffered_lats, self.buffered_lons)
        elif dataset in ["cmip6", "cmip"]:
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
            # print(self.env_vars)
            raster, _ = cmipper_file_ops.find_intersecting_cmip(
                self.env_vars,
                lats=self.buffered_lats,
                lons=self.buffered_lons,
                # lats=self.buffered_lats,
                # lons=self.buffered_lons,
                year_range=self.year_range_to_include,
                levs=self.levs,
                cmip6_data_dir=Path(
                    config.cmip6_data_dir
                    ) / self.config_info["source_id"] / self.config_info["member_id"] / "regridded",
            )
            if raster:
                return raster
            else:
                raise ValueError("No intersecting CMIP6 files found.")
        elif dataset == "new":
            return ds
        else:
            raise ValueError(f"Dataset {dataset} not recognised.")

    def get_resampled_raster(self, raster, dataset="cmip"):
        self.resolution = spatial_data.process_resolution_input(
            self.resolution, self.resolution_unit
        )
        print(f"\tresampling dataset to {self.resolution} degree(s) resolution...\n")

        if dataset in ["unep", "unep_wcmc", "gdcr", "unep_coral_presence"]:
            # count number of coral-containing cells in region
            resample_method = Resampling.sum
        else:
            resample_method = Resampling.bilinear

        # fetching more data than required necessary to avoid missing values at edge

        # print(self.buffered_lats, self.buffered_lons)

        # LATS = [-32, 0]
        # LONS = [130, 170]

        return rio_absolute_resample(
            # raster.sel(latitude=slice(*LATS), longitude=slice(*LONS)),
            raster.sel(latitude=slice(*self.buffered_lats), longitude=slice(*self.buffered_lons)),
            lat_resolution=self.resolution,
            lon_resolution=self.resolution,
            lat_range=self.lats,
            lon_range=self.lons,
            resample_method=resample_method,
            project_first=True,
        )

    def get_spatially_buffered_raster(self, raster):
        print("\tapplying spatial buffering...")
        # print(raster)
        # print(self.spatial_buffer)
        return apply_fill_loess(raster, nx=self.spatial_buffer, ny=self.spatial_buffer)

    def process_timeseries_to_static(self, raster):
        # TODO: add processing for other timseries datasets
        # if dataset in ["cmip6", "cmip"]:
        print("\tcalculating statistics for static ML model(s)...")
        static_ds = ml_processing.calculate_statistics(
            raster,
            vars=self.env_vars,
            years_window=self.year_range_to_include,
        )
        return static_ds
        # else:
        #     raise ValueError(
        #         f"Dataset {self.dataset} not recognised as appropriate timeseries."
        #     )
        # return static_ds

    def return_raster(self, dataset=None, ds=None):
        # order of operations decided to minimise unnecessarily intensive processing while
        # preserving information
        if dataset in ["unep", "unep_wcmc", "gdcr", "unep_coral_presence"]:
            dtype = np.float32  # necessary for nan values when resampling

        if dataset == "new":
            processed_raster = spatial_data.process_xa_d(
                self.get_raw_raster(dataset, ds=ds)).astype(dtype)
        else:
            processed_raster = spatial_data.process_xa_d(
                self.get_raw_raster(dataset).astype(dtype)
            )
        # this shouldn't be necessary (process_xa_d included in cmip download, but made a change to remove time_bnds
        # since then)

        if self.ds_type == "static" and dataset in ["cmip6", "cmip"]:
            processed_raster = self.process_timeseries_to_static(processed_raster)

        # return processed_raster
        if self.spatial_buffer:
            # if "spatial_ref" in processed_raster.coords:
            #     processed_raster.reset_coords('spatial_ref', drop=True)
            # TODO: figure out how best to calculate buffer: and whether to buffer at all
            buffered_raster = spatial_data.process_xa_d(
                self.get_spatially_buffered_raster(processed_raster)
            )
        resampled_raster = self.get_resampled_raster(buffered_raster, dataset=dataset)

        # if dataset in ["unep", "unep_wcmc", "gdcr", "unep_coral_presence"]:
        #     # normalise to coral cover
        #     lat_res, lon_res = abs(resampled_raster.rio.resolution()[0]), abs(resampled_raster.rio.resolution()[1])
        #     cell_area = lat_res * lon_res   # in degrees
        #     cell_area = cell_area * 110e3 * 110e3 * np.cos(np.mean(resampled_raster.latitude) * np.pi / 180)  # in m^2
        #     resampled_raster = 30**2 * resampled_raster / cell_area     # area of single observation of unep

        return resampled_raster


def calc_scale_factor(initial_resolution: float, final_resolution: float) -> float:
    """
    Calculate the scale factor between initial and final resolutions.

    Args:
        initial_resolution: The initial resolution.
        final_resolution: The final resolution.

    Returns:
        The scale factor between initial and final resolutions.
    """
    return initial_resolution / final_resolution


def scaled_width_height(
    lat_scale_factor: float,
    lon_scale_factor: float,
    initial_width: int,
    initial_height: int,
) -> tuple[int, int]:
    """
    Calculate the scaled width and height based on the scale factors and initial dimensions.

    Args:
        lat_scale_factor: The scale factor for latitude.
        lon_scale_factor: The scale factor for longitude.
        initial_width: The initial width.
        initial_height: The initial height.

    Returns:
        A tuple containing the scaled width and height.
    """
    return round(initial_width * lon_scale_factor), round(
        initial_height * lat_scale_factor
    )


def resample_rasterio(
    xa_d: xa.DataArray | xa.Dataset,
    lat_resolution: float,
    lon_resolution: float,
    method: Resampling = Resampling.bilinear,
) -> xa.DataArray | xa.Dataset:
    """
    Resample a raster using rasterio.

    Args:
        xa_d: The input raster as a xarray DataArray.
        lat_resolution: The desired latitude resolution.
        lon_resolution: The desired longitude resolution.
        method: The resampling method to use. Defaults to Resampling.bilinear.

    Returns:
        The resampled raster as a xarray DataArray.
    """
    lat_scale_factor = calc_scale_factor(
        abs(xa_d.rio.resolution()[0]), lat_resolution
    )
    lon_scale_factor = calc_scale_factor(
        abs(xa_d.rio.resolution()[1]), lon_resolution
    )

    new_width, new_height = scaled_width_height(
        lat_scale_factor, lon_scale_factor, xa_d.rio.width, xa_d.rio.height
    )
    if new_width == 0 or new_height == 0:
        raise ValueError(
            f"Cannot resample to 0 width or height. width: {new_width}, height: {new_height}"
        )
    # don't want nans slipping by without realising, and leading to ValueError when resampling first
    # if isinstance(xa_d, xa.DataArray):
    #     xa_d.rio.write_nodata(np.nan, inplace=True)

    return spatial_data.process_xa_d(xa_d.rio.reproject(
        xa_d.rio.crs,
        shape=(new_height, new_width),
        resampling=method,
    ))


def rasterize_geodf(
    geo_df: gpd.geodataframe,
    resolution: float = 1.0,
    all_touched: bool = False,
    merge_alg: MergeAlg = MergeAlg.replace,
    crs: str = "4326",
) -> np.ndarray:
    """
    Rasterizes a GeoDataFrame into a numpy array.

    Args:
        geo_df (gpd.geodataframe): The GeoDataFrame to be rasterized.
        resolution (float, optional): The resolution of the raster. Defaults to 1.0.
        all_touched (bool, optional): Whether to consider all pixels touched by the geometry. Defaults to True.
        merge_alg (MergeAlg, optional): The merge algorithm to use. Defaults to MergeAlg.replace.
        crs (str, optional): The coordinate reference system of the raster. Defaults to "4326".

    Returns:
        np.ndarray: The rasterized numpy array.

    N.B. some work done in reef_cover.ipynb to replace merge_alg and all_touched
    """
    xmin, ymin, xmax, ymax, width, height = lat_lon_vals_from_geo_df(
        geo_df, resolution
    )
    # Create the transform based on the extent and resolution
    transform = rasterio.transform.from_bounds(xmin, ymin, xmax, ymax, width, height)
    transform.crs = rasterio.crs.CRS.from_epsg(crs)

    return featuresio.rasterize(
        [(shape, 1) for shape in geo_df["geometry"]],
        out_shape=(height, width),
        transform=transform,
        fill=0,     # not nan since integer, and want to have negative target (0)
        all_touched=all_touched,    # argument to be made that this doesn't best reflect suitability
        dtype=rasterio.uint16,  # updated since was reaching upper limit of uint8
        merge_alg=merge_alg,    # ignore overlapping polygons
    )


def lat_lon_vals_from_geo_df(geo_df: gpd.geodataframe, resolution: float = 1.0):
    # Calculate the extent in degrees from bounds of geometry objects
    lon_min, lat_min, lon_max, lat_max = geo_df["geometry"].total_bounds
    # Calculate the width and height of the raster in pixels based on the extent and resolution
    width = int((lon_max - lon_min) / resolution)
    height = int((lat_max - lat_min) / resolution)

    return lon_min, lat_min, lon_max, lat_max, width, height


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

    # print(buffered_dataset.coords)

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
            if isinstance(buffered_dataset.time.values, np.datetime64):
                # grid = pyxarray.Grid2D(var_data)
                # filled = loess(grid, nx=nx, ny=ny)
                # buffered_data_vars[var_name].loc[dict(time=0)] = filled.T
                print("need multiple time values for now")
                continue    # TODO: writing when single value of time
            else:
                for t in tqdm(
                    buffered_dataset.time,
                    desc=f"Processing timesteps of variable '{var_name}'",
                    leave=False,
                    position=1,
                ):  # buffer each timestep
                    grid = pyxarray.Grid2D(var_data.sel(time=t))
                    filled = loess(grid, nx=nx, ny=ny)
                    buffered_data_vars[var_name].loc[dict(time=t)] = filled.T
        elif utils.check_var_has_coords(
            var_data, ["latitude", "longitude"]
        ):  # if dataset has latitude, longitude only
            grid = pyxarray.Grid2D(
                var_data.astype("float64"), geodetic=False
            )  # type required since loess can't handle uint8
            filled = loess(grid, nx=nx, ny=ny)

            # Transpose filled array if necessary: slightly hacky
            if filled.shape != buffered_dataset[var_name].shape:
                print("Transposing filled array to match the original shape.")
                filled = np.transpose(filled)

            # Check if dimensions match before updating
            if filled.shape != buffered_dataset[var_name].shape:
                raise ValueError(
                    f"""Dimensions of filled array do not match the original data array. Filled shape: {filled.shape},
                    Original shape: {buffered_dataset[var_name].shape}""")

            buffered_dataset.update(
                {var_name: (sorted(buffered_dataset[var_name].dims), filled)}
            )
        else:
            print(
                f"""Variable must have at least 'latitude', and 'longitude' coordinates to be spatially padded.
                \nVariable '{var_name}' has {var_data.coords}. Skipping..."""
            )

    return buffered_dataset


# def resample_rasterio(
#     rio_xa: xa.DataArray,
#     lat_resolution: float,
#     lon_resolution: float,
#     method: Resampling = Resampling.bilinear,
#     n_threads: int = 4,
# ) -> xa.DataArray:
#     """
#     Resample a raster using rasterio.

#     Args:
#         rio_xa: The input raster as a xarray DataArray.
#         lat_resolution: The desired latitude resolution.
#         lon_resolution: The desired longitude resolution.
#         method: The resampling method to use. Defaults to Resampling.bilinear.

#     Returns:
#         The resampled raster as a xarray DataArray.
#     """
#     lat_scale_factor = calc_scale_factor(
#         abs(rio_xa.rio.resolution()[0]), lat_resolution
#     )
#     lon_scale_factor = calc_scale_factor(
#         abs(rio_xa.rio.resolution()[1]), lon_resolution
#     )

#     new_width, new_height = scaled_width_height(
#         lat_scale_factor, lon_scale_factor, rio_xa.rio.width, rio_xa.rio.height
#     )
#     if new_width == 0 or new_height == 0:
#         raise ValueError(
#             f"Cannot resample to 0 width or height. width: {new_width}, height: {new_height}"
#         )

#     # rio_xa.rio.write_nodata(np.nan, inplace=True)
#     return rio_xa.rio.reproject(
#         rio_xa.rio.crs,
#         shape=(new_height, new_width),
#         resampling=method,
#         n_threads=n_threads
#     )


def resample_process_rasterio(
    rio_xa: xa.DataArray,
    lat_resolution: float,
    lon_resolution: float,
    method: Resampling = Resampling.bilinear,
) -> xa.DataArray:
    """
    Process and resample a raster using rasterio.

    Args:
        rio_xa: The input raster as a xarray DataArray.
        lat_resolution: The desired latitude resolution.
        lon_resolution: The desired longitude resolution.
        method: The resampling method to use. Defaults to Resampling.bilinear.

    Returns:
        The processed and resampled raster as a xarray DataArray.
    """
    return spatial_data.process_xa_d(
        resample_rasterio(rio_xa, lat_resolution, lon_resolution, method)
    )


def rio_resample_to_other(
    xa_d: xa.DataArray,
    other_xa_d: xa.DataArray,
    resampling_method: Resampling = Resampling.bilinear,
    project_first: bool = True
) -> xa.DataArray:
    """
    Resample a raster to match the resolution of another raster.

    Args:
        xa_d: The input raster as a xarray DataArray.
        other_xa_d: The other raster to match the resolution to.
        method: The resampling method to use. Defaults to Resampling.bilinear.

    Returns:
        The resampled raster as a xarray DataArray.
    """
    final_lat_resolution = abs(other_xa_d.rio.resolution()[0])
    final_lon_resolution = abs(other_xa_d.rio.resolution()[1])

    if project_first:
        # reproject first to match gridding
        reprojected = spatial_data.process_xa_d(
            xa_d.rio.reproject_match(other_xa_d, resampling=resampling_method)
        )
        # resample to correct resolution
        return resample_process_rasterio(
            reprojected, final_lat_resolution, final_lon_resolution, resampling_method
        )
    else:
        resampled = resample_process_rasterio(
            xa_d, final_lat_resolution, final_lon_resolution, resampling_method
        )
        return spatial_data.process_xa_d(
            resampled.rio.reproject_match(other_xa_d, resampling=resampling_method)
        )


def rio_absolute_resample(
    xa_d: xa.DataArray | xa.Dataset,
    lat_resolution: float,
    lon_resolution: float,
    lat_range: list[float] = None,
    lon_range: list[float] = None,
    resample_method: Resampling = Resampling.bilinear,
    project_first: bool = True

):
    if not xa_d.rio.crs:
        xa_d = xa_d.rio.write_crs("EPSG:4326")
        print("Resampling pipeline: written raster CRS to EPSG:4326")

    common_dataset = xa.Dataset(
        coords={
            "latitude": (
                ["latitude"],
                np.arange(lat_range[0], lat_range[1] + lat_resolution, lat_resolution),
            ),
            "longitude": (
                ["longitude"],
                np.arange(lon_range[0], lon_range[1] + lon_resolution, lon_resolution),
            ),
        }
    ).rio.write_crs(xa_d.rio.crs)
    return spatial_data.process_xa_d(
        rio_resample_to_other(xa_d, common_dataset, resample_method, project_first=project_first))


def old_resample_xa_d(
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
    return xa_d.sel(
        latitude=slice(*lat_range), longitude=slice(*lon_range)
    ).interp_like(common_dataset, method=resample_method)

    # current_resolution = utils.get_resolution(xa_d)
    # # if upsampling, interpolate
    # if resolution < current_resolution:
    #     return xa_d.sel(
    #         latitude=slice(*lat_range), longitude=slice(*lon_range)
    #     ).interp_like(common_dataset, method=resample_method)
    # # if downsampling, coarsen
    # else:
    #     return coarsen_xa_d(
    #         xa_d.sel(latitude=slice(*lat_range), longitude=slice(*lon_range)),
    #         resolution,
    #         resample_method,
    #     )


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

    # xa_d = xa_d.chunk({"latitude": 100, "longitude": 100})

    regridder = xe.Regridder(
        xa_d.astype("float64", order="C"),
        target_grid.chunk({"y": 100, "x": 100, "y_b": 100, "x_b": 100}),
        method=resampled_method,
        parallel=True,
    )

    # return spatial_data.process_xa_d(regridder(xa_d))
    return process_xesmf_regridded(regridder(xa_d.astype("float64", order="C")))


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


# DEPRECATED – SPLIT UP AND USING RIOXARRAY
# def spatially_combine_xa_d_list(
#     xa_d_list: list[xa.DataArray | xa.Dataset],
#     lat_range: list[float],
#     lon_range: list[float],
#     resolution: float,
#     resample_method: str = "linear",
# ) -> xa.Dataset:
#     """
#     Resample and merge a list of xarray DataArrays or Datasets to a common extent and resolution.

#     Args:
#         xa_d_list (list[xa.DataArray | xa.Dataset]): List of xarray DataArrays or Datasets to resample and merge.
#         lat_range (list[float]): Latitude range of the common extent.
#         lon_range (list[float]): Longitude range of the common extent.
#         resolution (float): Resolution of the common extent.
#         resample_method (str, optional): Resampling method to use. Defaults to "linear".

#     Returns:
#         xa.Dataset: Dataset containing the resampled and merged DataArrays or Datasets.

#     N.B. resample_method can take the following values for 1D interpolation:
#         - "linear": Linear interpolation.
#         - "nearest": Nearest-neighbor interpolation.
#         - "zero": Piecewise-constant interpolation.
#         - "slinear": Spline interpolation of order 1.
#         - "quadratic": Spline interpolation of order 2.
#         - "cubic": Spline interpolation of order 3.
#     and these for n-d interpolation:
#         - "linear": Linear interpolation.
#         - "nearest": Nearest-neighbor interpolation.
#     """

#     # Create a new dataset with the common extent and resolution
#     common_dataset = xa.Dataset(
#         coords={
#             "latitude": (["latitude"], np.arange(*np.sort(lat_range), resolution)),
#             "longitude": (
#                 ["longitude"],
#                 np.arange(*np.sort(lon_range), resolution),
#             ),
#         }
#     )

#     # Iterate through input datasets, resample, and merge into the common dataset
#     for input_ds in tqdm(
#         xa_d_list, desc=f"resampling and merging {len(xa_d_list)} datasets"
#     ):
#         # Resample the input dataset to the common resolution and extent using bilinear interpolation
#         resampled_dataset = input_ds.interp(
#             latitude=common_dataset["latitude"].sel(
#                 latitude=slice(min(lat_range), max(lat_range))
#             ),
#             longitude=common_dataset["longitude"].sel(
#                 longitude=slice(min(lon_range), max(lon_range))
#             ),
#             method=resample_method,
#         )

#         # Merge the resampled dataset into the common dataset
#         common_dataset = xa.merge(
#             [common_dataset, resampled_dataset], compat="no_conflicts"
#         )

#     return common_dataset


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
    for i in range(100):
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

        # print(f"{i} depth mask", depth_mask_lims)
        # print("pos_neg_ratio", non_zero_ratio)

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

# general
import numpy as np
import pandas as pd

# spatial
import xarray as xa
import rasterio
import haversine
from rasterio import features as rfeatures

# from scipy.ndimage import binary_dilation, generic_filter

# file ops
from tqdm import tqdm
from pathlib import Path

# custom
from coralshift.utils import file_ops, utils


def spatial_predictions_from_data(y: pd.Series, predictions: np.ndarray) -> pd.Series:
    predictions = pd.Series(predictions, index=y.index)

    # join dataframes on index
    merged = pd.concat(
        [pd.Series(y, name="label"), pd.Series(predictions, name="predictions")],
        axis=1,
    )

    return spatially_reform_data(merged)


def spatially_reform_data(df, resolution: float = None):
    """
    Reformats the input dataframe by adding missing rows for latitude and longitude values based on the specified
    resolution.

    Args:
        df (pd.DataFrame or pd.Series): The input dataframe to be reformatted.
        resolution (float, optional): The resolution for latitude and longitude values

    Returns:
        pd.DataFrame or pd.Series: The reformatted dataframe with added rows for missing latitude and longitude values.
    """
    if not resolution:
        lat_res_calc = abs(np.diff(df.sort_index().index.get_level_values("latitude")))
        lon_res_calc = abs(np.diff(df.sort_index().index.get_level_values("longitude")))
        resolution = np.min([np.min(lat_res_calc[lat_res_calc != 0]), np.min(lon_res_calc[lon_res_calc != 0])]).round(5)

    lat_spacing, lon_spacing = resolution, resolution

    # Create arrays of latitude and longitude values using inferred spacing
    latitudes = np.arange(
        df.index.get_level_values("latitude").min(),
        df.index.get_level_values("latitude").max() + lat_spacing,
        lat_spacing,
    )
    longitudes = np.arange(
        df.index.get_level_values("longitude").min(),
        df.index.get_level_values("longitude").max() + lon_spacing,
        lon_spacing,
    )
    # prepare index from (unrounded) index of 
    index = pd.MultiIndex.from_product(
        [latitudes, longitudes], names=["approx_latitude", "approx_longitude"]
    )
    if isinstance(df, pd.Series):
        new_df = df.reindex(index, fill_value=np.nan).to_frame()
    else:
        new_df = df.reindex(index, fill_value=np.nan)

    new_df["latitude"] = new_df.index.get_level_values("approx_latitude").round(5)
    new_df["longitude"] = new_df.index.get_level_values("approx_longitude").round(5)

    df = df.reset_index(inplace=False)
    df["latitude"] = df["latitude"].round(5)
    df["longitude"] = df["longitude"].round(5)
    # return df
    new_df = new_df.reset_index(inplace=False)
    new_df.set_index(["latitude", "longitude"], inplace=True)
    new_df.drop(columns=["approx_latitude", "approx_longitude"], inplace=True)
    # return df
    df.set_index(["latitude", "longitude"], inplace=True)

    return df.combine_first(new_df).to_xarray().sortby(["latitude", "longitude"])

# def spatially_reform_data(df, resolution: float = None):
#     """TODO: docstring"""

#     return df.to_xarray().sortby(["latitude", "longitude"])


def resample_xarray_to_target(
    xa_d: xa.DataArray | xa.Dataset,
    target_resolution_d: float,
    lat_lims: tuple[float] = None,
    lon_lims: tuple[float] = None,
    method=rasterio.enums.Resampling.bilinear,
    name: str = " ",
    crs: str = "EPSG:4326",
) -> xa.Dataset:
    """
    Upsamples an xarray DataArray or Dataset to a target resolution.

    Parameters
    ----------
        xa_d (xarray.DataArray or xarray.Dataset): The input xarray object to upsample.
        target_resolution (float): The target resolution in degrees.

    Returns
    -------
        xarray.Dataset: The upsampled dataset.

    Notes
    -----
        - The function resamples the input xarray object by coarsening it to a target resolution.
        - The current implementation supports upsampling along latitude and longitude dimensions only.
        - The function calculates the degree resolution of the input dataset and scales it to match the target
        resolution.
        - The resampling is performed by coarsening the dataset using a mean operation.
    """
    # if coordinate ranges not specified, infer from present xa_d
    if not (lat_lims and lon_lims):
        lat_lims = min_max_of_coords(xa_d, "latitude")
        lon_lims = min_max_of_coords(xa_d, "longitude")

    target_xa_d = generate_dummy_xa(
        target_resolution_d, lat_lims=lat_lims, lon_lims=lon_lims
    )

    return resample_xa_d_to_other(xa_d, target_xa_d, name=name)
    # resampled_xa_das_dict = {}
    # for xa_da in tqdm(xa_das, desc="Resampling xarray DataArrays"):
    #     # xa_resampled = resample_xa_d_to_other(xa_da, dummy_xa, name=xa_da.name)
    #     xa_resampled = spatial_data.resample_xa_d_to_other(xa_da, target_xa_d)
    #     resampled_xa_das_dict[xa_da.name] = xa_resampled
    # # N.B. not perfect at getting starts/ends matching up It works fine when not trying to match the coords of another
    # # write crs
    # xa_d.rio.write_crs("EPSG:4326", inplace=True)
    # return process_xa_d(
    #     xa_d.rio.reproject("EPSG:4326", target_resolution, resampling=method)
    # )

    # TODO: enable flexible upsampling by time also
    # lat_lims = xarray_coord_limits(xa_d, "latitude")
    # lon_lims = xarray_coord_limits(xa_d, "longitude")

    # dummy_xa = generate_dummy_xa(target_resolution, lat_lims, lon_lims)

    # return resample_xa_d_to_other(xa_d, dummy_xa, method=method, name=name)


def generate_dummy_xa(
    resolution: float, lat_lims: tuple[float], lon_lims: tuple[float]
) -> xa.DataArray:
    """
    Generate a dummy xarray DataArray with consistent coordinate order.

    Parameters
    ----------
    resolution (float): The resolution in degrees for the latitude and longitude values.
    lat_lims (tuple[float]): The latitude limits as a tuple of (min, max) values.
    lon_lims (tuple[float]): The longitude limits as a tuple of (min, max) values.

    Returns
    -------
    xa.DataArray: The generated dummy xarray DataArray.
    """
    # Generate latitude and longitude values
    lat_vals = np.arange(min(lat_lims), max(lat_lims) + resolution / 2, resolution)
    lon_vals = np.arange(min(lon_lims), max(lon_lims) + resolution / 2, resolution)

    # Create a DataArray filled with NaN values
    data = np.full((len(lat_vals), len(lon_vals)), np.nan)

    # Create xarray DataArray
    xa_dummy = (
        # xa.DataArray(data, coords=[lat_vals, lon_vals], dims=["latitude", "longitude"])
        xa.DataArray(data, coords=[lat_vals, lon_vals], dims=["y", "x"])
        .rio.write_crs("EPSG:4326")
        .rename("dummy")
    )

    # Reindex the DataArray to ensure consistent coordinate order
    xa_dummy = xa_dummy.reindex(
        latitude=sorted(xa_dummy.y), longitude=sorted(xa_dummy.x)
    )

    return xa_dummy


def downsample_interp(
    xa_d: xa.DataArray | xa.Dataset,
    lat_vals: tuple[float],
    lon_vals: tuple[float],
    interp_method: str = "linear",
) -> xa.DataArray | xa.Dataset:
    """
    Downsample and interpolate an xarray DataArray or Dataset along the latitude and longitude dimensions.

    Parameters
    ----------
        xa_d (xa.DataArray | xa.Dataset): The xarray DataArray or Dataset to be downsampled and interpolated.
        lat_vals (tuple[float]): A tuple of latitude values to downsample and interpolate to.
        lon_vals (tuple[float]): A tuple of longitude values to downsample and interpolate to.
        interp_method (str): The interpolation method to use. Supported methods are 'linear', 'nearest', 'cubic'.

    Returns
    -------
        xa.DataArray | xa.Dataset: The downsampled and interpolated xarray DataArray or Dataset.
    """
    return xa_d.interp(latitude=lat_vals, longitude=lon_vals, method=interp_method)


def resample_xa_d_to_other(
    xa_d: xa.DataArray | xa.Dataset,
    target_xa_d: xa.DataArray | xa.Dataset,
    method=rasterio.enums.Resampling.bilinear,
    name: str = None,
    var_name: str = None,
) -> xa.DataArray | xa.Dataset:
    """
    Upsample a given xarray DataArray or Dataset to match the spatial characteristics of another DataArray or Dataset.

    Parameters
    ----------
    xa_d (xarray.DataArray or xarray.Dataset): The xarray object to upsample.
    target_xa_d (xarray.DataArray or xarray.Dataset): The target xarray object with the desired spatial characteristics
        (e.g., resolution, extent).
    method (rasterio.enums.Resampling, optional): The resampling method to use during upsampling. Default is
        rasterio.enums.Resampling.bilinear.
    name (str, optional): The name to assign to the upsampled DataArray or Dataset. If not provided, the original
        name is used.

    Returns
    -------
    xarray.DataArray or xarray.Dataset: The upsampled xarray object with matching spatial characteristics as the target.
    """
    # xa_da = file_ops.extract_variable(xa_d, var_name)
    # xa_resampled = xa_da.rio.reproject_match(target_xa_d, resampling=method)
    xa_d = xa_d.rename({"latitude": "y", "longitude": "x"})
    xa_resampled = xa_d.rio.reproject_match(target_xa_d, resampling=method)

    xa_processed = process_xa_d(xa_resampled)
    if not name == " ":
        return xa_processed.rename(name)
    else:
        return xa_processed


def upsample_xarray_by_factor(
    xa_array: xa.DataArray | xa.Dataset, factors: dict
) -> xa.DataArray:
    """Upsamples (decreases resolution) of an xarray.DataArray object by a given factor along each dimension.

    Parameters
    ----------
        xa_array (xarray.DataArray): xarray object to upsample.
        factors (dict): dictionary containing the factor by which each coord should be upsampled e.g. {"x": 10} will
        lower the resolution by a factor of 10.

    Returns
    -------
        xarray.DataArray: upsampled xarray object, obtained by coarsening the input object by the given factor
            along each dimension and then taking the mean of the resulting blocks.
    """
    coarse_array = xa_array.coarsen(dim=factors, boundary="pad").mean()
    # # resizing like this removes any advantage of upsampling in the first place...
    # resized_coarse_array = xa.DataArray(np.resize(coarse_array, xa_array.shape), coords=xa_array.coords)

    existing_xa_array = xa_array.copy(deep=True)
    # existing_xa_array.data = resized_coarse_array.data
    existing_xa_array.data = coarse_array.data
    return existing_xa_array


def upsample_and_save_xa_a(
    save_dir: Path | str,
    xa_d: xa.Dataset | xa.DataArray,
    target_resolution_d: float = None,
    target_xa_d: xa.DataArray | xa.Dataset = None,
    method=None,
    name: str = None,
    crs: str = "EPSG:4326",
) -> tuple[xa.DataArray, str]:
    """
    Upsample an xarray dataset or data array and save the upsampled data.

    Parameters
    ----------
        save_dir (Path or str): The directory where the upsampled data will be saved.
        xa_d (xa.Dataset or xa.DataArray): The xarray dataset or data array to be upsampled.
        target_resolution_d (float, optional): The target spatial resolution for the upsampled data.
        target_xa_d (xa.DataArray or xa.Dataset, optional): The target xarray dataset or data array with which to match
        the resolution.
        method: The method to be used for upsampling.
        name (str, optional): The name to be given to the upsampled data.

    Returns
    -------
        tuple[xa.DataArray, str]: A tuple containing the filepath and the saved array of the upsampled data.
    """
    # TODO: shift upsampling to after checking file exists
    xa_d.rio.write_crs(crs, inplace=True)
    if target_resolution_d:
        xa_upsampled = resample_xarray_to_target(
            xa_d, target_resolution=target_resolution_d
        )
    else:
        xa_upsampled = resample_xa_d_to_other(xa_d, target_xa_d)
        # generate target resolution for filename
        target_resolution_d = degrees_to_distances(
            np.mean(calculate_spatial_resolution(target_xa_d))
        )

    if not name:
        name = xa_d.name

    filename = f"{name}_{target_resolution_d:.05}d"
    filepath, array = file_ops.save_nc(
        save_dir, filename, xa_upsampled, return_array=True
    )

    return filepath, array


# def process_xa_array(
#     xa_array: xa.DataArray,
#     coords_to_drop: list[str] = None,
#     coords_to_rename: dict = {"x": "longitude", "y": "latitude"},
#     # resolution: float = None,
#     # shape: tuple[int] = None,
#     verbose: bool = True,
# ) -> xa.DataArray:
#     """Process the given xarray DataArray by dropping and renaming specified coordinates.

#     Parameters
#     ----------
#         xa_array (xa.DataArray): xarray DataArray to be processed.
#         coords_to_drop (list[str]): list of coordinates to be dropped from the DataArray.
#         coords_to_rename (dict, optional): dictionary of coordinates to be renamed in the DataArray.
#             Defaults to {"x": "longitude", "y": "latitude"}.
#         resolution (float, optional): Output resolution of the upsampled DataArray, in the same units as the input.
#             Defaults to None.
#         shape (tuple, optional): Shape of the output DataArray as (height, width). If specified, overrides the
#             resolution parameter. Defaults to None.
#         verbose (bool, optional): if True, print information about the remaining coordinates in the DataArray.
#             Defaults to True.

#     Returns
#     -------
#         xa.DataArray: The processed xarray DataArray.
#     """
#     if not coords_to_drop:
#         # drop specified coordinates
#         xa_array = xa_array.drop_vars(coords_to_drop)
#     # rename specified coordinates
#     xa_array = xa_array.rename(coords_to_rename)

#     # if resolution or shape:
#     #     upsample_xa_array(xa_array, resolution=resolution, shape=shape)
#     if verbose:
#         # show info about remaining coords
#         print(xa_array.coords)

#     return xa_array.sortby()


# def process_xa_arrays_in_dict(
#     xa_array_dict: dict,
#     coords_to_drop: list[str],
#     coords_to_rename: dict = {"x": "longitude", "y": "latitude"},
#     resolution: float = None,
#     shape: tuple[int] = None,
# ) -> dict:
#     """Process multiple xarray DataArrays stored in a dictionary by dropping and renaming specified coordinates.

#     Parameters
#     ----------
#         xa_array_dict (dict): A dictionary of xarray DataArrays to be processed: keys = filename,
# values = xa.DataArray.
#         coords_to_drop (list[str]): A list of coordinate fields to be dropped from each DataArray.
#         coords_to_rename (dict): A dictionary of coordinate fields to be renamed in each DataArray.

#     Returns
#     -------
#         dict: A dictionary containing the processed xarray DataArrays.
#     """
#     processed_dict = {}
#     print(
#         f"Processing xa_arrays in dictionary. Dropping {coords_to_drop}, renaming {coords_to_rename.keys()}."
#     )
#     for name, xa_array in tqdm(xa_array_dict.items(), desc="processing xarray: "):
#         processed_dict[name] = process_xa_array(
#             xa_array, coords_to_drop, coords_to_rename, resolution, shape, verbose=False
#         )

#     return processed_dict


def tifs_to_xa_array_dict(tif_paths: list[Path] | list[str]) -> dict:
    """Given a list of file paths to GeoTIFF files, open each file and create a dictionary where each key is the
    filename of the GeoTIFF file (without its directory path or extension) and each value is the contents of the file as
    a np.ndarray array.

    Parameters
    ----------
        tif_paths (list[Path] | list[str]): list of file paths to GeoTIFF files. Paths can be provided as either Path
            objects or strings.

    Returns
    -------
        A dictionary where each key is the filename of a GeoTIFF file and each value is the contents of the file as a
        np.ndarray.
    """
    xa_array_dict = {}
    for tif in tif_paths:
        filename = tif.stem
        tif_array = file_ops.tif_to_xa_array(tif)
        xa_array_dict[filename] = tif_array.rename(filename)

    return xa_array_dict


def return_min_of_coord(xa_array: xa.DataArray, coord: str) -> float:
    """Returns the minimum value of the specified coordinate field in the given xarray DataArray.

    Parameters
    ----------
        xa_array (xa.DataArray): The xarray DataArray to search for the minimum value of the coordinate field.
        coord (str): The name of the coordinate field to find the minimum value of.

    Returns
    -------
        float: The minimum value of the specified coordinate field as a float.
    """
    return float(xa_array[coord].min().values)


def return_max_of_coord(xa_array: xa.DataArray, coord: str) -> float:
    """Returns the maximum value of the specified coordinate field in the given xarray DataArray.

    Parameters
    ----------
        xa_array (xa.DataArray): The xarray DataArray to search for the maximum value of the coordinate field.
        coord (str): The name of the coordinate field to find the maximum value of.

    Returns
    -------
        float: The maximum value of the specified coordinate field as a float.
    """
    return float(xa_array[coord].max().values)


def min_max_of_coords(
    xa_array: xa.DataArray | xa.Dataset, coord: str
) -> tuple[float, float]:
    """Returns the minimum and maximum values of the specified coordinate field in the given xarray DataArray.

    Parameters
    ----------
        xa_array (xa.DataArray): xarray DataArray to search for the minimum and maximum values of the coordinate field.
        coord (str): name of the coordinate field to find the minimum and maximum values of.

    Returns
    -------
        tuple[float, float]: tuple containing minimum and maximum values of the specified coordinate field as floats.
    """
    return return_min_of_coord(xa_array, coord), return_max_of_coord(xa_array, coord)


def min_max_of_all_spatial_coords(
    xa_ds_list: list[xa.DataArray, xa.Dataset]
) -> tuple[tuple[float]]:
    lats, lons = [], []
    for xa_d in xa_ds_list:
        lats.append(min_max_of_coords(xa_d, "latitude"))
        lons.append(min_max_of_coords(xa_d, "longitude"))

    return (np.min(lats), np.max(lats)), (np.min(lons), np.max(lons))


# def return_pixels_closest_to_value(
#     array: np.ndarray,
#     central_value: float,
#     tolerance: float = 0.5,
#     buffer_pixels: int = 10,
#     bathymetry_only: bool = True,
# ) -> np.ndarray:
#     """Returns a 1D array of all the pixels in the input array that are closest to a specified central value within a
#     given tolerance and within a pixel buffer zone.

#     Parameters
#     ----------
#         array (np.ndarray): The input array of pixel values.
#         central_value (float): The central value to which the pixels should be compared.
#         tolerance (float, optional): The tolerance within which the pixels are considered to be "close" to the central
#             value. Defaults to 0.5.
#         buffer_pixels (int, optional): The size of the buffer zone around the pixels. Defaults to 10.
#         bathymetry_only (bool, optional): Whether to only consider bathymetric data, i.e., values less than zero.
#             Defaults to True.

#     Returns
#     -------
#         np.ndarray: A 1D array of all the pixels in the input array that are closest to the specified central value
#         within the given tolerance and within the pixel buffer zone.
#     """
#     binary = np.isclose(array, central_value, atol=0.5)
#     # morphological dilation operation
#     dilated = binary_dilation(binary, iterations=buffer_pixels)

#     array_vals = array[dilated]
#     # if specifying only bathymetric data
#     if bathymetry_only:
#         array_vals = array_vals[array_vals < 0]

#     # return only non-zero values as 1d array
#     return array_vals[np.nonzero(array_vals)]


# def tif_to_xarray(tif_path: Path | str, renamed: str = None) -> xa.DataArray:
#     """
#     Convert a TIFF file to an xarray.DataArray object.

#     Parameters
#     ----------
#         tif_path (Path or str): Path to the TIFF file.
#         renamed (str, optional): Name to assign to the DataArray. If not provided, the original name will be used.

#     Returns
#     -------∂aArray: The converted DataArray object.
#     """
#     xa_da = rio.open_rasterio(rasterio.open(tif_path))
#     if renamed:
#         return process_xa_d(xa_da.rename(renamed))
#     else:
#         return process_xa_d(xa_da)


# def return_distance_closest_to_value(
#     array: np.ndarray,
#     central_value: float,
#     tolerance: float = 0.5,
#     buffer_distance: float = 300,
#     distance_per_pixel: float = 30,
#     bathymetry_only: bool = True,
# ) -> np.ndarray:
#     """Wrapper for return_pixels_closest_to_value() allowing specification by distance from thresholded values rather
#     than number of pixels

#     Returns a 1D array of all the pixels in the input array that are closest to a specified central value within a
#     given tolerance and within a distance buffer zone.

#     Parameters
#     ----------
#         array (np.ndarray): The input array of pixel values.
#         central_value (float): The central value to which the pixels should be compared.
#         tolerance (float, optional): The tolerance within which the pixels are considered to be "close" to the central
#             value. Defaults to 0.5.
#         buffer_distance (float, optional): The size of the buffer zone around the pixels. Defaults to 300.
#         bathymetry_only (bool, optional): Whether to only consider bathymetric data, i.e., values less than zero.
#             Defaults to True.

#     Returns
#     -------
#         np.ndarray: A 1D array of all the pixels in the input array that are closest to the specified central value
#         within the given tolerance and within the distance buffer zone.
#     """
#     buffer_pixels = buffer_distance / distance_per_pixel
#     return return_pixels_closest_to_value(
#         array, central_value, tolerance, buffer_pixels, bathymetry_only
#     )


# def upsample_xa_array(
#     xa_array: xa.DataArray, resolution: float = 1 / 12, shape: tuple = None
# ) -> xa.DataArray:
#     """Upsamples the resolution of a DataArray using rioxarray's 'reproject' functionality: reprojecting it onto a
# lower
#     resolution and/or differently-sized grid

#     Parameters
#     ----------
#     xa_array (xa.DataArray): Input DataArray to upsample.
#     resolution (float, optional): Output resolution of the upsampled DataArray, in the same units as the input.
#         Defaults to 1/12.
#     shape (tuple, optional): Shape of the output DataArray as (height, width). If specified, overrides the resolution
#         parameter.

#     Returns
#     -------
#     xa.DataArray: The upsampled DataArray
#     """

#     if not shape:
#         upsampled_array = xa_array.rio.reproject(
#             xa_array.rio.crs, resolution=resolution
#         )
#     else:
#         upsampled_array = xa_array.rio.reproject(xa_array.rio.crs, shape=shape)

#     return upsampled_array


# def upsample_dict_of_xa_arrays(
#     xa_dict: dict, resolution: float = 1 / 12, shape: tuple[int, int] = None
# ) -> dict:
#     """Upsamples the resolution of each DataArray in a dictionary and returns the upsampled dictionary.

#     Parameters
#     ----------
#     xa_dict (dict): Dictionary containing the input DataArrays.
#     resolution (float, optional): Output resolution of the upsampled DataArrays, in the same units as the input.
#         Defaults to 1/12.
#     shape (tuple, optional): Shape of the output DataArrays as (height, width). If specified, overrides the resolution
#         parameter.

#     Returns
#     -------
#     dict: Dictionary containing the upsampled DataArrays with corresponding keys as array_name_upsampled names.
#     TODO: add in check to ensure that upsampling rather than downsampling?
#     """
#     upsampled_dict = {}
#     print(f"Upsampling {xa_dict.keys()} from dictionary to {resolution}{shape}.")
#     for name, array in tqdm(xa_dict.items(), desc="processing xarray dict: "):
#         upsampled_name = "_".join(
#             (file_ops.remove_suffix(name), "upsampled", "{0:.3g}".format(resolution))
#         )
#         upsampled_array = upsample_xa_array(array, resolution, shape)
#         upsampled_dict[upsampled_name] = upsampled_array

#     return upsampled_dict


def xarray_coord_limits(xa_array: xa.Dataset, dim: str) -> tuple[float]:
    """Compute the minimum and maximum values for a coordinate dimension of an xarray dataset.

    Parameters
    ----------
        xa_array (xa.Dataset): input xarray dataset.
        dim (str): coordinate name.

    Returns
    -------
        tuple[float]: minimum and maximum values of the coordinate

    """
    min, max = float(xa_array[dim].min()), float(xa_array[dim].max())
    return (min, max)


def dict_xarray_coord_limits(xa_array: xa.Dataset) -> dict:
    """Compute the minimum and maximum values for each coordinate dimension of an xarray dataset and assign to a dict
    # TODO: rename generate_dict_xa_coord_limits
    Parameters
    ----------
        xa_array (xa.Dataset): input xarray dataset.

    Returns
    -------
        dict: dict object where keys are the names of the cooordinate and values are a tuple of the minimum and maximum
        values of the input xarray dataset.
    """
    lims_dict = {}
    for dim in xa_array.dims:
        lims_dict[dim] = xarray_coord_limits(xa_array, dim)

    return lims_dict


def rasterize_shapely_df(
    df: pd.DataFrame,
    class_col: str,
    shapes_col: str = "geometry",
    resolution: float = 1,
    all_touched: bool = True,
) -> np.ndarray:
    """Rasterizes a pandas DataFrame containing Shapely geometries.

    Parameters
    ----------
    df (pd.DataFrame): The input pandas DataFrame.
    class_col (str): The name of the column in `df` with the classes.
    shapes_col (str, optional): The name of the column in `df` with the Shapely geometries. Default is "geometry".
    resolution (float, optional) The resolution of the output raster. Default is 1.
    all_touched (bool, optional): Whether to consider all pixels touched by geometries or just their centroids. Default
        is True.

    Returns
    -------
    np.ndarray: A numpy array with the rasterized
    """
    # create empty raster grid
    xmin, ymin, xmax, ymax = df.total_bounds  # takes ages with large dfs
    width = int(np.ceil((xmax - xmin) / resolution))
    height = int(np.ceil((ymax - ymin) / resolution))
    # affine transform (handles projection)
    transform = rasterio.Affine(resolution, 0, xmin, 0, -resolution, ymax)
    raster = np.zeros((height, width))

    print("Rasterizing pandas DataFrame")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        class_value = row[class_col]
        shapes = [
            (row[shapes_col], 1)
        ]  # the second value (1) represents the value to assign to the raster cell
        rasterized = rfeatures.rasterize(
            shapes=shapes,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            all_touched=False,
            merge_alg=rasterio.enums.MergeAlg.replace,
        )
        raster[rasterized == 1] = int(class_value)

    return raster, (xmin, ymin, xmax, ymax)


def generate_lat_lon_arrays(
    lat_bounds: tuple[float], lon_bounds: tuple[float], resolution
) -> tuple[np.ndarray]:
    """Generates latitude and longitude arrays based on input bounds and resolution.

    Parameters
    ----------
    lat_bounds (tuple[float]): A tuple with the latitude bounds (bottom, top).
    lon_bounds (tuple[float]): A tuple with the longitude bounds (left, right).
    resolution (float): The resolution of the output arrays.

    Returns
    -------
    tuple[np.ndarray]: A tuple with the latitude and longitude arrays.
    """
    # xarray requires first coordinate in vertical direction to be the topmost
    # TODO: this could cause issues with coords in other quartiles
    if lat_bounds[0] < lat_bounds[1]:
        lats_list = np.arange(lat_bounds[1], lat_bounds[0], -resolution)
    else:
        lats_list = np.arange(lat_bounds[0], lat_bounds[1], resolution)

    lons_list = np.arange(lon_bounds[0], lon_bounds[1], resolution)

    return lats_list, lons_list


def xa_array_from_raster(
    raster: np.ndarray,
    lat_bounds: tuple[float],
    lon_bounds: tuple[float],
    resolution: float = 1 / 12,
    crs_tag: str = "epsg:4326",
    xa_name: str = "unnamed",
) -> xa.DataArray:
    """Creates an xarray DataArray from a raster numpy array.

    Parameters
    ----------
    raster (np.ndarray): A numpy array with the raster
    lat_bounds (tuple[float]): A tuple with the latitude bounds (bottom, top).
    lon_bounds (tuple[float]): A tuple with the longitude bounds (left, right).
    resolution (float, optional): The resolution of the output arrays. Default is 1/12.
    crs_tag (str, optional): The coordinate reference system (CRS) tag for the output array. Default is "epsg:4326".

    Returns
    -------
    xa.DataArray: An xarray DataArray with the raster data.
    """

    """Provide latitude and longitudes as (bottom_lat, top_lat) and (left_lon, right_lon) respectively. Will want to
    plot as if viewed on flat globe"""
    # todo automate calculation of resolution
    lats, lons = generate_lat_lon_arrays(
        (lat_bounds[0], lat_bounds[1]), (lon_bounds[0], lon_bounds[1]), resolution
    )
    coords_dict = {"latitude": lats, "longitude": lons}

    array = xa.DataArray(raster, coords_dict, name=xa_name)
    array.rio.write_crs(crs_tag, inplace=True)

    return array


def generate_raster_xa(
    xa_ds: xa.Dataset,
    resolution=1 / 12,
    class_col="class_val",
    xa_name: str = "",
    all_touched=True,
    filepath: Path | str = None,
) -> xa.Dataset:
    """Generate a raster from an xarray Dataset using rasterio and save it to disk as a netCDF file.

    Parameters
    ----------
    xa_ds (xarray.Dataset): The xarray Dataset containing the data to rasterize.
    resolution (float, optional): The resolution (in units of the input data) to use for the raster, by default 1/12.
    class_col (str, optional) The name of the column in `xa_ds` containing the class labels, by default "class_val".
    xa_name (str, optional): The name to use for the output xarray DataArray, by default an empty string.
    all_touched (bool, optional):  Whether to rasterize all pixels touched by the geometry or only those whose center is
        within the geometry, by default True.
    filepath (Path or str, optional): The path where the generated raster should be saved as a netCDF file, by default
        None.

    Returns
    -------
    xarray.Dataset: The xarray Dataset containing the generated raster.
    """
    # generate raster array
    raster_values, (xmin, ymin, xmax, ymax) = rasterize_shapely_df(
        xa_ds, class_col=class_col, resolution=resolution, all_touched=all_touched
    )
    raster_xa = xa_array_from_raster(
        raster_values, (ymin, ymax), (xmin, xmax), resolution=1 / 12, xa_name=xa_name
    )

    filepath = file_ops.add_suffix_if_necessary(filepath, ".nc")
    # if a path provided, save raster to that location
    if filepath:
        filepath = file_ops.add_suffix_if_necessary(filepath, ".nc")
        raster_xa.to_netcdf(filepath)
        print(f"{filepath} created.")

    return raster_xa


def check_nc_exists_generate_raster_xa(
    dir_path: Path | str,
    filename: str,
    xa_ds: xa.Dataset = None,
    resolution: float = None,
    horizontal_distance: float = None,
    class_col: str = None,
    xa_name: str = None,
    all_touched: bool = None,
) -> xa.DataArray | xa.Dataset:
    """Check if a raster file with the given filename already exists in the directory. If not, generate the raster file
    using the given xarray dataset, and save it to the directory with the given filename.

    Parameters
    ----------
    dir_path (Path | str): The directory path where the raster file should be saved.
    filename (str): The name of the raster file to be saved.
    xa_ds (xa.Dataset): The xarray dataset to be converted to a raster file.
    resolution (float): The resolution of the raster file to be generated.
    class_col (str): The column name in the xarray dataset that contains the classification data.
    xa_name (str): The name of the xarray dataset to be used in the raster file.
    all_touched (bool): If True, all pixels touched by geometries will be burned in. If False (default), only pixels
        whose center is within the polygon or that are selected by Bresenham's line algorithm will be burned in.

    Returns
    -------
    None
        If the raster file already exists, print a message indicating that the file exists and no files were written.
        If the raster file does not exist, generates the raster file and saves it to the directory with the given
        filename. Either way, returns raster
    """
    filepath = file_ops.add_suffix_if_necessary(Path(dir_path) / filename, ".nc")
    # if specifying resolution as horizontal distance rather than degrees lat/lon, calculate approximate degrees
    if horizontal_distance:
        resolution = distance_to_degrees(horizontal_distance)

    if not file_ops.check_file_exists(
        dir_path=dir_path, filename=filename, suffix=".nc"
    ):
        raster_xa = generate_raster_xa(
            xa_ds, resolution, class_col, xa_name, all_touched, filepath
        )
        return raster_xa
    else:
        print(f"{filename} already exists in {dir_path}. No new files written.")
        return xa.open_dataset(filepath)


def process_resolution_input(resolution: float, unit: str = "m") -> float:
    """
    Convert the input resolution from the specified unit to degrees or return the resolution as is.

    Parameters
    ----------
        resolution (float): The resolution value to be converted.
        unit (str, optional): The unit of the resolution. Valid options are 'm', 'metres', 'metre'
                              for meters, and 'd', 'degrees', 'degree' for degrees.
                              Defaults to 'm'.

    Returns
    -------
        float: The converted resolution in degrees or the original resolution if the unit is 'd' or 'degrees'.

    """
    if unit.lower() in ["km", "kilometers"]:
        _, _, degree_resolution = distance_to_degrees(resolution * 1000)
    elif unit.lower() in ["m", "metres", "metre"]:
        _, _, degree_resolution = distance_to_degrees(resolution)
    elif unit.lower() in ["d", "degrees", "degree"]:
        degree_resolution = resolution
    else:
        raise ValueError(f"Unit string {unit} not recognised.")

    return degree_resolution


def degrees_to_distances(
    target_lat_res: float,
    target_lon_res: float = None,
    approx_lat: float = -18,
    approx_lon: float = 145,
) -> tuple[float]:
    """Converts target latitude and longitude resolutions from degrees to distances (in meters).

    Parameters
    ----------
        target_lat_res (float): The target latitude resolution in degrees.
        target_lon_res (float, optional): The target longitude resolution in degrees.
            If not specified, the longitude resolution will be assumed to be the same as the latitude resolution.
        approx_lat (float, optional): The approximate latitude coordinate.
            It is used as a reference for distance calculations. Default is -18.
        approx_lon (float, optional): The approximate longitude coordinate.
            It is used as a reference for distance calculations. Default is 145.

    Returns
    -------
        tuple[float]: A tuple containing the converted distances in meters
        (latitude distance, longitude distance, mean distance).

    Notes
    -----
        - It uses the haversine formula to calculate the distance between two coordinates on a sphere.
        - By default, the function assumes an approximate latitude of -18 and an approximate longitude of 145.
        - If only the latitude resolution is specified, the function assumes the longitude resolution is the same.
    """
    start_coord = (approx_lat, approx_lon)
    lat_end_coord = (approx_lat + target_lat_res, approx_lon)
    # if both lat and lon resolutions specified
    if target_lon_res:
        lon_end_coord = (approx_lat, approx_lon + target_lon_res)
    else:
        lon_end_coord = (approx_lat, approx_lon + target_lat_res)

    return (
        haversine.haversine(start_coord, lat_end_coord, unit=haversine.Unit.METERS),
        haversine.haversine(start_coord, lon_end_coord, unit=haversine.Unit.METERS),
        np.mean(
            (
                haversine.haversine(
                    start_coord, lat_end_coord, unit=haversine.Unit.METERS
                ),
                haversine.haversine(
                    start_coord, lon_end_coord, unit=haversine.Unit.METERS
                ),
            )
        ),
    )


def distance_to_degrees(
    distance_lat: float,
    distance_lon: float = None,
    approx_lat: float = -18,
    approx_lon: float = 145,
) -> tuple[float, float, float]:
    """Converts a distance in meters to the corresponding distance in degrees, given an approximate location on Earth.
    # TODO: check function: returning hypotenuse seems way too small
    Parameters
    ----------
    distance_lat (float): The distance in meters along the latitude direction.
    distance_lon (float, optional): The distance in meters along the longitude direction. If not provided, it is
        assumed to be the same as `distance_lat`.
    approx_lat (float, optional): The approximate latitude of the location in degrees. Defaults to -18.0.
    approx_lon (float, optional): The approximate longitude of the location in degrees. Defaults to 145.0.

    Returns
    -------
    tuple[float, float, float]: A tuple containing the corresponding distance in degrees:
        - The distance in degrees along the latitude direction.
        - The distance in degrees along the longitude direction.
        - The hypotenuse distance encapsulating the difference in both latitude and longitude.
    """
    # if distance_lon not provided, assume the same as distance_lat
    if not distance_lon:
        distance_lon = distance_lat

    degrees_lat = haversine.inverse_haversine(
        (approx_lat, approx_lon),
        distance_lat,
        haversine.Direction.SOUTH,
        unit=haversine.Unit.METERS,
    )

    degrees_lon = haversine.inverse_haversine(
        (approx_lat, approx_lon),
        distance_lon,
        haversine.Direction.WEST,
        unit=haversine.Unit.METERS,
    )

    # calculate the coordinates 'distance' meters to the southwest (chosen to give measure of both lat and lon)
    av_distance = (distance_lat + distance_lon) / 2
    (lat_deg, lon_deg) = haversine.inverse_haversine(
        (approx_lat, approx_lon),
        av_distance,
        haversine.Direction.SOUTHWEST,
        unit=haversine.Unit.METERS,
    )
    delta_lat, delta_lon = abs(lat_deg - approx_lat), abs(lon_deg - approx_lon)
    # return hypotenuse (encapsulates difference in both lat and lon)
    return (
        np.subtract((approx_lat, approx_lon), degrees_lat)[0],
        np.subtract((approx_lat, approx_lon), degrees_lon)[1],
        np.hypot(delta_lat, delta_lon),
    )


def filter_strings(
    str_list: list[str], exclude: list[str] = ["latitude", "longitude", "depth", "time"]
) -> list[str]:
    """Filters a list of strings to exclude those contained in a second list of excluded strings.

    Parameters
    ----------
    str_list (list[str]): A list of strings to filter.
    exclude (list[str], optional): A list of strings to exclude. Defaults to ["latitude", "longitude", "depth", "time"].

    Returns
    -------
    list[str]: A list of the filtered strings.
    """
    # Filter strings using list comprehension
    filtered_strings = [s for s in str_list if s not in exclude]
    # Return filtered strings
    return filtered_strings


def xa_ds_to_3d_numpy(
    xa_ds: xa.Dataset,
    exclude_vars: list[str] = ["latitude", "longitude", "depth", "time"],
) -> np.ndarray:
    """Convert an xarray dataset to a 3D numpy array.

    Parameters
    ----------
    xa_ds (xarray.Dataset): The xarray dataset to convert.
    exclude_vars (list[str], optional): A list of variable names to exclude from the conversion.
        Default is ["latitude", "longitude", "depth", "time"].

    Returns
    -------
    np.ndarray: The converted 3D numpy array.
    """
    # stack the dataset
    ds_stacked = xa_ds.stack(location=("latitude", "longitude"))

    array_list = []
    variables_to_read = filter_strings(list(xa_ds.variables), exclude_vars)
    for var in tqdm(
        variables_to_read, desc="converting xarray Dataset to numpy arrays: "
    ):
        vals = ds_stacked[var].values
        array_list.append(vals)

    # swap first and third columns. New shape: grid_cell_val x var x time
    return np.moveaxis(np.array(array_list), 2, 0)


def check_array_empty(array: xa.DataArray) -> bool:
    """Check if all values in an xarray DataArray are empty (i.e., all values are null/NaN).

    Parameters
    ----------
        array (xa.DataArray): xarray DataArray to be checked

    Returns
    -------
        bool: True if all values are 0/NaN; False otherwise
    """
    vals = array.values
    # replace nans with zeros
    vals[np.isnan(vals)] = 0
    # return True if any non-zero elements; False otherwise
    if np.sum(vals) == 0:
        return True
    else:
        return False


def return_non_empty_vars(xa_array: xa.Dataset) -> list[str]:
    """Return a list of variable names in the given xarray dataset that have non-empty values.

    Parameters
    ----------
        xa_array (xr.Dataset): The xarray dataset to check for non-empty variables.

    Returns
    -------
        List[str]: A list of variable names with non-empty values.
    """
    non_empty_vars = []
    for var_name in xa_array.data_vars:
        if not check_array_empty(xa_array[var_name]):
            non_empty_vars.append(var_name)

    return non_empty_vars


def date_from_dt(datetime: str | np.datetime64) -> str:
    """Converts a datetime string to a string with just the date.

    Parameters
    ----------
    datetime (str): string representing the datetime.

    Returns
    -------
    str: A string representing the date extracted from the input datetime string.

    Example
    -------
        date_from_dt('2023-05-11 14:25:00')
        Output: '2023-05-11'
    """
    return str(pd.to_datetime(datetime).date())


# def resample_dataarray(xa_da: xa.DataArray, start_end_freq: tuple[str]) -> xa.DataArray:
#     """Resample a dataarray's time coordinate to a new frequency and date range

#     TODO: fix this. Weird non-use of time_index, and makes up new values rather than selecting those closest
#     """
#     start_date, end_date, freq = start_end_freq
#     # Convert time coordinate to pandas DatetimeIndex
#     # time_index = pd.DatetimeIndex(xa_da.time.values)
#     # Create a new DatetimeIndex with the desired frequency and date range (will fetch closest value if not exact)
#     new_index = pd.date_range(start=start_date, end=end_date, freq=freq)
#     # Resample the dataarray's time coordinate to the new frequency
#     # resampled_da = xa_da.reindex(time=new_index).ffill(dim='time')
#     resampled_da = xa_da.reindex(time=new_index)

#     # problem with not matching what's already there
#     return resampled_da, freq


def get_variable_values(xa_ds: xa.Dataset, var_name: str) -> list[np.ndarray]:
    """Returns a list of ndarrays containing the values of the given variable for each timestep in the 'time'
    coordinate of the xarray Dataset.

    Parameters
    ----------
    xa_ds (xa.Dataset): The xarray Dataset to extract the variable from.
    var_name (str): The name of the variable to extract.

    Returns
    -------
    list: A list of ndarrays containing the values of the variable for each timestep in the 'time' coordinate.
    """
    values_list = []
    for time in xa_ds.time:
        values_list.append(xa_ds[var_name].sel(time=time).values)
    return values_list


# def buffer_nans(array: np.ndarray, num_pixels: float = 1) -> np.ndarray:
#     """Buffer nan values in a 2D array by taking the mean of valid neighbors.

#     Parameters
#     ----------
#         array (ndarray): Input 2D array with NaN values representing land.
#         num_pixels (int, optional): Buffer num_pixels in pixels. Defaults to 1.

#     Returns
#     -------
#         np.ndarray: Buffered array with NaN values replaced by the mean of valid neighbors.
#     """

#     def nan_sweeper(values: np.ndarray):
#         """Custom function to sweep NaN values and calculate the mean of valid neighbors.

#         Parameters
#         ----------
#             values (np.ndarray): 1D array representing the neighborhood of an element.

#         Returns
#         -------
#             float: Mean value of valid neighbors or the central element if not NaN.
#         """
#         central_value = values[len(values) // 2]
#         # check if central element of kernel is nan
#         if np.isnan(central_value):
#             if np.isnan(values).all():
#                 return central_value
#             # extract valid (non-nan values)
#             valid_values = values[~(np.isnan(values))]
#             # and return the mean of these values, to be assigned to rest of the buffer kernel
#             return np.mean(valid_values)
#         else:
#             return central_value

#     actual_num_pixels = num_pixels + 1
#     # call nan_sweeper on each element of "array"
#     # "constant" – array extended by filling all values beyond edge with same constant value, defined by cval
#     buffered_array = generic_filter(
#         array, nan_sweeper, size=actual_num_pixels, mode="constant", cval=np.nan
#     )
#     return buffered_array


# def buffer_nans(array: np.ndarray, num_pixels: int = 1) -> np.ndarray:
#     """Buffer NaN values in a 2D array by taking the mean of valid neighbors.

#     Parameters:
#         array (ndarray): Input 2D array with NaN values representing land.
#         num_pixels (int, optional): Number of pixels to buffer by. Defaults to 1.

#     Returns:
#         np.ndarray: Buffered array with NaN values replaced by the mean of valid neighbors.
#     """

#     def nan_sweeper(values):
#         # Check if the central element of the kernel is NaN
#         if np.isnan(values[len(values) // 2]):
#             # Extract valid (non-NaN) values
#             valid_values = values[~np.isnan(values)]
#             if len(valid_values) == 0:
#                 return np.nan  # All neighbors are NaN
#             else:
#                 return np.mean(valid_values)
#         else:
#             return values[len(values) // 2]

#     # kernel_size = 2 * num_pixels + 1  # Kernel size considering neighbors
#     kernel_size = num_pixels + 1
#     buffered_array = generic_filter(
#         array, nan_sweeper, size=kernel_size, mode="constant", cval=np.nan
#     )
#     return buffered_array


def buffer_nans(array: np.ndarray, num_pixels: int = 1) -> np.ndarray:
    """Buffer NaN values in a 2D array by taking the mean of valid neighbors.

    Parameters:
        array (ndarray): Input 2D array with NaN values representing land.
        num_pixels (int, optional): Number of pixels to buffer by. Defaults to 1.

    Returns:
        np.ndarray: Buffered array with NaN values replaced by the mean of valid neighbors.
    """
    from skimage.restoration import inpaint

    mask = np.isnan(array).astype(int)
    buffered_array = inpaint.inpaint_biharmonic(array, mask)
    return buffered_array

    # # Use Dask array instead of NumPy array for parallel computation
    # dask_array = da.from_array(array, chunks=array.shape)
    # buffered_array = da.map_blocks(
    #     buffer_nans,generate_and_add_gt_to_xa_d
    #     dask_array,
    #     size=size,
    #     dtype=float,
    #     drop_axis=0,  # Drop the first axis (the one processed by map_blocks)
    # )
    # return buffered_array.compute()  # Compute the result to get a NumPy array

    # for data_var in tqdm(
    #     filtered_vars, desc=f"Buffering variables by {buffer_size} pixel(s)"
    # ):
    #     buffered = xa.apply_ufunc(
    #     buffer_nans,
    #     xa_ds[data_var].chunk(),
    #     # xa_ds[data_var],
    #     input_core_dims=[[]],
    #     output_core_dims=[[]],
    #     kwargs={"size": buffer_size},
    #     dask="parallelized",  # Enable parallel computation with Dask
    #     output_dtypes=[float]
    # )
    # buffered_ds[data_var] = buffered


def filter_out_nans(X_with_nans: np.ndarray, y_with_nans: np.ndarray) -> np.ndarray:
    """Filters out NaN values from 3d input arrays (columns, rows, and depths: columns contain entirely NaN values are
    removed; while rows and depths which contain any NaN values are removed. Resulting X array has a shape of
    (num_samples, seq_length, num_params).

    Parameters
    ----------
        X_with_nans (np.ndarray): Input array containing NaN values.
            It must have a shape of (num_samples, num_params, seq_length).
        y_with_nans (np.ndarray): Target array corresponding to X_with_nans.
            It must have a shape of (num_samples,).

    Returns
    -------
        tuple[ndarray]: A tuple containing the filtered X array and the filtered y array.
    """
    # must be in shape (num_samples, num_params, seq_length)

    # filter out columns that contain entirely NaN values
    # boolean mask indicating which columns to keep
    col_mask = ~np.all(np.isnan(X_with_nans), axis=(0, 2))
    # keep only the columns that don't contain entirely NaN values
    masked_cols = X_with_nans[:, col_mask, :]

    # filter out all rows which contain any NaN values
    # boolean mask indicating which rows to keep
    row_mask = ~np.any(np.isnan(masked_cols), axis=1)
    # keep only the rows that don't contain any NaN values
    masked_cols_rows = masked_cols[row_mask[:, 0], :, :]

    # filter out all depths which contain any NaN values
    # boolean mask indicating which depths to keep
    depth_mask = ~np.any(np.isnan(masked_cols_rows), axis=(0, 1))
    # keep only the depths that don't contain any NaN values
    X = masked_cols_rows[:, :, depth_mask]
    # swap axes so shape (num_samples, seq_length, num_params)
    X = np.swapaxes(X, 1, 2)

    y = y_with_nans[row_mask[:, 0]]

    return X, y


def chunk_as_necessary(
    xa_d: xa.DataArray | xa.Dataset, chunk_dict: dict
) -> xa.DataArray | xa.Dataset:
    """
    Chunk the input xarray DataArray or Dataset along the specified dimensions according to the given chunk dictionary.

    Parameters
    -----------
        xa_d (xa.DataArray or xa.Dataset): The xarray DataArray or Dataset to be chunked.
        chunk_dict (dict): A dictionary specifying the chunk size for each dimension.
                           The keys are the dimension names, and the values are the desired chunk sizes.
                           If a dimension is not present in the dictionary or its value is None,
                           that dimension will not be chunked.

    Returns
    -------
        xa.DataArray or xa.Dataset: The chunked xarray DataArray or Dataset.

    """
    # set up to chunk relevant coordinates
    dims_to_chunk = {dim: chunk_dict.get(dim, None) for dim in xa_d.dims}
    return xa_d.chunk(dims_to_chunk)


def process_xa_d(
    xa_d: xa.Dataset | xa.DataArray,
    rename_lat_lon_grids: bool = False,
    rename_mapping: dict = {
        "lat": "latitude",
        "lon": "longitude",
        "y": "latitude",
        "x": "longitude",
        "i": "longitude",
        "j": "latitude",
        "lev": "depth",
    },
    squeeze_coords: str | list[str] = None,
    # chunk_dict: dict = {"latitude": 100, "longitude": 100, "time": 100},
    crs: str = "EPSG:4326",
):
    """
    Process the input xarray Dataset or DataArray by standardizing coordinate names, squeezing dimensions,
    chunking along specified dimensions, and sorting coordinates.

    Parameters
    ----------
        xa_d (xa.Dataset or xa.DataArray): The xarray Dataset or DataArray to be processed.
        rename_mapping (dict, optional): A dictionary specifying the mapping for coordinate renaming.
            The keys are the existing coordinate names, and the values are the desired names.
            Defaults to a mapping that standardizes common coordinate names.
        squeeze_coords (str or list of str, optional): The coordinates to squeeze by removing size-1 dimensions.
                                                      Defaults to ['band'].
        chunk_dict (dict, optional): A dictionary specifying the chunk size for each dimension.
                                     The keys are the dimension names, and the values are the desired chunk sizes.
                                     Defaults to {'latitude': 100, 'longitude': 100, 'time': 100}.

    Returns
    -------
        xa.Dataset or xa.DataArray: The processed xarray Dataset or DataArray.

    """
    temp_xa_d = xa_d.copy()

    if rename_lat_lon_grids:
        temp_xa_d = temp_xa_d.rename(
            {"latitude": "latitude_grid", "longitude": "longitude_grid"}
        )

    for coord, new_coord in rename_mapping.items():
        if new_coord not in temp_xa_d.coords and coord in temp_xa_d.coords:
            temp_xa_d = temp_xa_d.rename({coord: new_coord})
    # temp_xa_d = xa_d.rename(
    #     {coord: rename_mapping.get(coord, coord) for coord in xa_d.coords}
    # )
    if "band" in temp_xa_d.dims:
        temp_xa_d = temp_xa_d.squeeze("band")
    if squeeze_coords:
        temp_xa_d = temp_xa_d.squeeze(squeeze_coords)

    if "time" in temp_xa_d.dims:
        temp_xa_d = temp_xa_d.transpose("time", "latitude", "longitude", ...)
    else:
        temp_xa_d = temp_xa_d.transpose("latitude", "longitude")

    if "grid_mapping" in temp_xa_d.attrs:
        del temp_xa_d.attrs["grid_mapping"]

    # drop variables which will never be variables
    # TODO: add as argument with default
    drop_vars = ["time_bnds", "lat_bnds", "lon_bnds"]

    if isinstance(temp_xa_d, xa.Dataset):
        temp_xa_d = temp_xa_d.drop_vars(
            [var for var in drop_vars if var in temp_xa_d.variables]
        )

    temp_xa_d = utils.mask_above_threshold(temp_xa_d, threshold=1e10)

    # sort coordinate dimensions in order time (if time present), latitude, longitude
    temp_xa_d = temp_xa_d.sortby(list(temp_xa_d.dims))

    # add crs
    # temp_xa_d.rio.write_crs(crs, inplace=True)    # was messing with fill_loess
    # if chunk_dict is not None:
    #     temp_xa_d = chunk_as_necessary(temp_xa_d, chunk_dict)
    # sort coords by ascending values
    return temp_xa_d.sortby(list(temp_xa_d.dims))


def xa_region_from_coord_bounds(xa_d, coord_bounds_dict):
    """Selects a region from a xarray DataArray based on coordinate bounds.

    Parameters
    ----------
        xa_d (xarray.DataArray): The input DataArray to select a region from.
        coord_bounds_dict (dict): A dictionary containing the coordinate bounds for each dimension. The keys should
            correspond to the dimension names in `xa_d`, and the values should be tuples or lists representing the lower
            and upper bounds for each dimension.

    Returns
    -------
        xarray.DataArray: The selected region from `xa_d` based on the provided coordinate bounds.
    """
    slice_dict = {}
    for k, v in coord_bounds_dict.items():
        slice_dict[k] = slice(min(v), max(v))

    return xa_d.sel(slice_dict)


def xa_region_from_window(
    xa_d: xa.Dataset | xa.DataArray,
    lat_lon_starts: tuple = (0, 0),
    coord_lengths: tuple[int, int] = (6, 6),
    coord_range: tuple[float, float] = None,
):
    """Sample a spatial batch from an xarray Dataset or DataArray.
    # TODO: check this works
    Parameters
    ----------
        xa_d (xarray.Dataset or xarray.DataArray): The input Dataset or DataArray to sample a spatial batch from.
        lat_lon_starts (tuple, optional): A tuple containing the starting latitude and longitude coordinates.
            Default is (0, 0).
        coord_lengths (tuple[int, int], optional): A tuple specifying the number of latitude and longitude cells to
            include in the spatial batch. Default is (6, 6).
        coord_range (tuple[float, float], optional): A tuple containing the latitude and longitude range in degrees. If
            provided, `coord_lengths` will be ignored, and the range will be used to determine the spatial batch size.
            Default is None.

    Returns
    -------
        xarray.Dataset or xarray.DataArray: The sampled spatial batch from `xa_d` based on the provided parameters.
    """
    lat_start, lon_start = lat_lon_starts[0], lat_lon_starts[1]
    if coord_range:
        lat_cells, lon_cells = coord_range[0], coord_range[1]
        lat_range = (lat_start, lat_start + lat_cells)
        lon_range = (lon_start, lon_start + lon_cells)
        subsample = xa_d.sel(
            {
                "latitude": slice(min(lat_range), max(lat_range)),
                "longitude": slice(min(lon_range), max(lon_range)),
            }
        )
    else:
        subsample = xa_d.isel(
            {
                "latitude": slice(lat_start, coord_lengths[0]),
                "longitude": slice(lon_start, coord_lengths[1]),
            }
        )
    return subsample


def sample_spatial_batch(
    xa_d: xa.Dataset,
    lat_lon_starts: tuple = (0, 0),
    coord_lengths: tuple[int, int] = (6, 6),
    coord_range: tuple[float] = None,
    variables: list[str] = None,
) -> np.ndarray:
    """Sample a spatial batch from an xarray Dataset.
    Parameters
    ----------
    xa_d (xa.Dataset): The input xarray Dataset.
    lat_lon_starts (tuple): Tuple specifying the starting latitude and longitude indices of the batch.
    coord_range (tuple[int, int]): Tuple specifying the dimensions (number of cells) of the spatial window.
    coord_range (tuple[float], optional): Tuple specifying the latitude and longitude range (in degrees) of the spatial
        window. If provided, it overrides the coord_range parameter.
    variables (list[str], optional): List of variable names to include in the spatial batch. If None, includes all
        variables.

    Returns
    -------
    np.ndarray: The sampled spatial batch as a NumPy array.
    """
    # if selection of variables specified
    if variables is not None:
        xa_d = xa_d[variables]

    subsample = xa_region_from_window(xa_d, lat_lon_starts, coord_lengths, coord_range)

    # # N.B. have to be careful when providing coordinate ranges for areas with negative coords. TODO: make universal
    # lat_start, lon_start = lat_lon_starts[0], lat_lon_starts[1]

    # if not coord_range:
    #     subsample = xa_d.isel(
    #         {
    #             "latitude": slice(lat_start, coord_range[0]),
    #             "longitude": slice(lon_start, coord_range[1]),
    #         }
    #     )
    # else:
    #     lat_cells, lon_cells = coord_range[0], coord_range[1]
    #     subsample = xa_d.sel(
    #         {
    #             "latitude": slice(lat_start, lat_start + lat_cells),
    #             "longitude": slice(lon_start, lon_start + lon_cells),
    #         }
    #     )

    lat_slice = subsample["latitude"].values
    lon_slice = subsample["longitude"].values
    time_slice = subsample["time"].values

    return subsample, {
        "latitude": lat_slice,
        "longitude": lon_slice,
        "time": time_slice,
    }


def process_xa_ds_for_ml(
    xa_ds: xa.Dataset,
    feature_vars: list[str] = None,
    gt_var: str = None,
    normalise: bool = True,
    onehot: bool = True,
) -> tuple[np.ndarray, ...]:
    """
    Process xarray Dataset for machine learning.

    Parameters
    ----------
    xa_ds : xa.Dataset
        The input xarray dataset.
    feature_vars : list[str], optional
        List of variable names to be used as features. Default is None.
    gt_var : str, optional
        The variable name for the ground truth. Default is None.
    normalise : bool, optional
        Flag indicating whether to normalize each variable between 0 and 1. Default is True.
    onehot : bool, optional
        Flag indicating whether to encode NaN values using the one-hot method. Default is True.

    Returns
    -------
    tuple[np.ndarray, ...]
        A tuple containing the feature array and ground truth array.
    """
    to_return = []
    if feature_vars is not None:
        # switch
        # assign features and convert to lat, lon to latxlon column

        Xs = spatial_array_to_column(xa_d_to_np_array(xa_ds[feature_vars]))

        # if normalise = True, normalise each variable between 0 and 1
        if normalise:
            Xs = normalise_3d_array(Xs)
        # remove columns containing only nans. TODO: enable removal of all nan dims
        nans_array = exclude_all_nan_dim(Xs, dim=1)

        # if encoding nans using onehot method
        if onehot:
            Xs = encode_nans_one_hot(nans_array)
        to_return.append(naive_nan_replacement(Xs))

    if gt_var:
        # assign ground truth and convert to column vector
        ys = spatial_array_to_column(xa_d_to_np_array(xa_ds[gt_var]))
        # ys = naive_nan_replacement(ys)

        # take single time slice (since broadcasted back through time)
        # ys = ys[:, 0]
        to_return.append(naive_nan_replacement(ys))

    return tuple(to_return)


def generate_patch(
    xa_ds: xa.DataArray | xa.Dataset,
    lat_lon_starts: tuple[float, float],
    coord_lengths: tuple[float, float] = None,
    coord_range: tuple[int, int] = None,
    feature_vars: list[str] = ["bottomT", "so", "mlotst", "uo", "vo", "zos", "thetao"],
    gt_var: str = "coral_algae_1-12_degree",
    normalise: bool = True,
    onehot: bool = True,
) -> tuple[np.ndarray, xa.Dataset | xa.DataArray, dict]:
    """Generate a patch for training or evaluation.
    Parameters
    ----------
    xa_ds (xa.Dataset): The input xarray dataset.
    lat_lon_starts (tuple): The starting latitude and longitude indices for sampling the patch.
    coord_lengths (tuple): The latitude and longitude range for sampling the patch.
    feature_vars (list[str], optional): List of variable names to be used as features.
        Default is ["bottomT", "so", "mlotst", "uo", "vo", "zos", "thetao"].
    gt_var (str, optional): The variable name for the ground truth. Default is "coral_algae_1-12_degree".
    normalise (bool, optional): Flag indicating whether to normalize each variable between 0 and 1. Default is True.
    onehot (bool, optional): Flag indicating whether to encode NaN values using the one-hot method. Default is True.

    Returns
    -------
    tuple[np.ndarray, xa.Dataset | xa.DataArray, dict]: A tuple containing the feature array, ground truth array,
        subsampled dataset, and latitude/longitude values.
    """
    subsample, lat_lon_vals_dict = sample_spatial_batch(
        process_xa_d(xa_ds),
        lat_lon_starts=lat_lon_starts,
        coord_lengths=coord_lengths,
        coord_range=coord_range,
    )

    output = process_xa_ds_for_ml(
        xa_ds=subsample,
        feature_vars=feature_vars,
        gt_var=gt_var,
        normalise=normalise,
        onehot=onehot,
    )

    return output, subsample, lat_lon_vals_dict


def subsample_to_array(
    xa_ds: xa.Dataset | xa.DataArray,
    lat_lon_starts: tuple,
    coord_lengths: tuple,
    variables: list[str],
) -> tuple[np.ndarray, xa.Dataset | xa.DataArray, dict]:
    """
    Subsample specific variables from an xarray dataset and convert them to a NumPy array.

    Parameters
    ----------
        xa_ds (xa.Dataset): The input xarray dataset.
        lat_lon_starts (tuple): The starting latitude and longitude indices for subsampling.
        coord_lengths (tuple): The latitude and longitude range for subsampling.
        variables (list[str]): List of variable names to subsample and convert.

    Returns
    -------
        tuple[np.ndarray, xa.Dataset | xa.DataArray, dict]: A tuple containing the subsampled array, subsampled dataset,
            and latitude/longitude values.
    """
    subsample, lat_lon_vals_dict = sample_spatial_batch(
        xa_ds[variables], lat_lon_starts=lat_lon_starts, coord_lengths=coord_lengths
    )
    return xa_d_to_np_array(subsample), subsample, lat_lon_vals_dict


def xa_d_to_np_array(xa_d: xa.Dataset | xa.DataArray) -> np.ndarray:
    """Converts an xarray dataset or data array to a NumPy array.

    Parameters
    ----------
        xa_d (xarray.Dataset or xarray.DataArray): The xarray dataset or data array to convert.

    Returns
    -------
        np.ndarray: The converted NumPy array.

    Raises
    ------
        TypeError: If the provided object is neither an xarray Dataset nor an xarray DataArray.
    """
    # if xa.DataArray
    if utils.is_type_or_list_of_type(xa_d, xa.DataArray):
        ds = xa_d.transpose("latitude", "longitude", ...)
        return np.array(ds.values)

    # else if dataset
    elif utils.is_type_or_list_of_type(xa_d, xa.Dataset):
        # transpose coordinates for consistency
        ds = xa_d.transpose("latitude", "longitude", ...)
        # send to array N.B. slow for larger datasets
        array = ds.to_array().values
        # reorder dimensions to (lat x lon x var x time)
        return np.moveaxis(array, 0, 3)
    else:
        return TypeError(
            "Object provided as argument was neither an xarray Dataset nor xarray DataArray."
        )


def naive_nan_replacement(array: np.ndarray, replacement: float = 0) -> np.ndarray:
    """Replace NaN values in a NumPy array with a specified replacement value.

    Parameters
    ----------
    array (np.ndarray): The input array.
    replacement (float, optional): The value to replace NaNs with. Default is 0.

    Returns
    -------
    np.ndarray: The array with NaN values replaced.
    """
    # replace nans with "replacement"
    array[np.isnan(array)] = 0
    return array


def exclude_all_nan_dim(array: np.ndarray, dim=int):
    """Exclude columns from a 2D or higher-dimensional array that contain only NaN values.

    Parameters
    ----------
    array (np.ndarray): The input array.
    dim (int): The dimension along which to check for NaN values (e.g., columns).

    Returns
    -------
    np.ndarray: The array with columns that contain only NaN values removed.
    """
    # TODO: check performance. Currently only able to hand columns (see generalisation comment)
    # filter out columns that contain entirely NaN values
    num_dims = len(array.shape)
    axes = tuple(set(np.arange(0, num_dims)) - {dim})

    dim_mask = ~np.all(
        np.isnan(array), axis=axes
    )  # boolean mask indicating which columns to keep
    # TODO: need to generalise this
    return array[
        :, dim_mask, :
    ]  # keep only the columns that don't contain entirely NaN values


def spatial_array_to_column(array: np.ndarray) -> np.ndarray:
    """Reshape the first two dimensions of a 3D NumPy array to a column vector.

    Parameters
    ----------
    array (np.ndarray): The input 3D NumPy array.

    Returns
    -------
    np.ndarray: The reshaped column vector.

    Examples
    --------
    array = np.random.rand(lat, lon, var, ...)
    column_vector = spatial_array_to_column(array)
    print(column_vector.shape)
    (lat x lon, var, ...)
    """
    array_shape = array.shape
    new_shape = (array_shape[0] * array_shape[1], *array.shape[2:])
    return np.reshape(array, new_shape)


# def exclude_all_nan_dim(array, dim):
#     # filter out columns that contain entirely NaN values
#     col_mask = ~np.all(np.isnan(nans_array), axis=(0,2)) # boolean mask indicating which columns to keep
#     return nans_array[:, col_mask, :] # keep only the columns that don't contain entirely NaN values

# exclude all

# exclude_all_nan_dim(nans_array, 1).shape


def reshape_from_ds_sample(xa_ds_sample: xa.Dataset, predicted: np.ndarray):
    """Reshapes the predicted array to the original dimensions of the xarray dataset sample (obtained from "latitude",
    "longitude", "time", and any "variables" dimensions of the sample).

    Parameters
    ----------
        xa_ds_sample (xarray.Dataset or xarray.DataArray): Sample from the original xarray dataset.
        predicted (ndarray or Tensor): Predicted array to be reshaped.

    Returns
    -------
        ndarray: Reshaped predicted array with the original dimensions.
    """
    # reshape to original dimensions
    original_shape = [xa_ds_sample.dims[d] for d in ["latitude", "longitude", "time"]]
    original_shape += [len(list(xa_ds_sample.data_vars))]
    original_shape = tuple(original_shape)

    return predicted.numpy().reshape(original_shape[:2])


def assign_prediction_to_ds(
    xa_ds: xa.Dataset,
    reshaped_pred: np.ndarray,
    lat_lon_dict: dict,
    new_var_name: str = "output",
) -> xa.Dataset:
    """Assigns the reshaped prediction array to a new variable "new_var_name" in the xarray dataset.

    Parameters
    ----------
        xa_ds (xarray.Dataset): Input xarray dataset.
        reshaped_pred (ndarray): Reshaped prediction array.
        lat_lon_dict (dict): Dictionary containing latitude and longitude coordinate arrays.
        new_var_name (str): Name for the new variable. Default is "output".

    Returns
    -------
        xarray.Dataset: Updated xarray dataset with the prediction assigned to a new variable.
    """
    # Create a new variable in the dataset using the output subset
    xa_ds[new_var_name] = xa.DataArray(
        reshaped_pred,
        dims=["latitude", "longitude"],
        coords={key: lat_lon_dict[key] for key in ["latitude", "longitude"]},
    )

    lat_start, lat_end = lat_lon_dict["latitude"].min(), lat_lon_dict["latitude"].max()
    lon_start, lon_end = (
        lat_lon_dict["longitude"].min(),
        lat_lon_dict["longitude"].max(),
    )

    # Set values outside the subset range to NaN

    xa_ds["output"] = xa_ds["output"].where(
        (xa_ds["output"].latitude >= lat_start)
        & (xa_ds["output"].latitude <= lat_end)
        & (xa_ds["output"].longitude >= lon_start)
        & (xa_ds["output"].longitude <= lon_end),
        np.nan,
    )
    return xa_ds


def normalise_3d_array(array: np.ndarray) -> np.ndarray:
    """Normalizes a 3D array between 0 and 1 along the sample and sequence dimensions by computing the min and max
    values for each variable along the sample and sequence dimensions.

    Parameters
        array (ndarray): Input 3D array.

    Returns:
        ndarray: Normalized 3D array.
    """
    # Compute the minimum and maximum values for each variable
    min_vals = np.nanmin(
        array, axis=(0, 1)
    )  # Minimum values along sample and seq dimensions
    max_vals = np.nanmax(
        array, axis=(0, 1)
    )  # Maximum values along sample and seq dimensions

    # Normalize the values for each variable between 0 and 1
    return np.divide((array - min_vals), (max_vals - min_vals))


def reformat_prediction(
    xa_ds: xa.Dataset,
    sample: xa.Dataset,
    predicted: np.ndarray,
    lat_lon_vals_dict: dict,
) -> xa.Dataset:
    """Reformats the predicted array and assigns it to the xarray dataset.

    Parameters
        xa_ds (xarray.Dataset): Input xarray dataset.
        sample (xarray.Dataset or xarray.DataArray): Sample from the original xarray dataset.
        predicted (ndarray or Tensor): Predicted array.
        lat_lon_vals_dict (dict): Dictionary containing latitude and longitude coordinate arrays.

    Returns:
        xarray.Dataset: Updated xarray dataset with the reformatted prediction assigned to a new variable.
    """
    reshaped_pred = reshape_from_ds_sample(sample, predicted)
    return assign_prediction_to_ds(xa_ds, reshaped_pred, lat_lon_vals_dict)


def spatially_buffer_timeseries(
    xa_ds: xa.Dataset,
    buffer_size: int = 3,
    exclude_vars: list[str] = ["spatial_ref", "coral_algae_1-12_degree"],
) -> xa.Dataset:
    """Applies a spatial buffer to each data variable in the xarray dataset.

    Parameters
        xa_ds (xarray.Dataset): Input xarray dataset.
        buffer_size (int): Buffer size in grid cells.
        exclude_vars (list[str]): List of variable names to exclude from buffering.

    Returns:
        xarray.Dataset: Xarray dataset with buffered data variables.
    """
    filtered_vars = [var for var in xa_ds.data_vars if var not in exclude_vars]

    buffered_ds = xa.Dataset()
    for data_var in tqdm(
        filtered_vars, desc=f"Buffering variables by {buffer_size} pixel(s)"
    ):
        buffered = xa.apply_ufunc(
            buffer_nans,
            xa_ds[data_var],
            input_core_dims=[[]],
            output_core_dims=[[]],
            kwargs={"num_pixels": buffer_size},
            dask="parallelized",
        )
        buffered_ds[data_var] = buffered

    return buffered_ds


def spatially_buffer_nc_file(nc_path: Path | str, buffer_size: int = 3):
    # TODO: specify distance buffer
    nc_path = Path(nc_path)
    buffered_name = nc_path.stem + f"_buffered_{buffer_size}_pixel"
    buffered_path = (nc_path.parent / buffered_name).with_suffix(".nc")

    # if buffered file doesn't already exist
    if not buffered_path.is_file():
        buffered_ds = spatially_buffer_timeseries(
            xa.open_dataset(nc_path), buffer_size=buffer_size
        )
        buffered_ds.to_netcdf(buffered_path)
    else:
        buffered_ds = xa.open_dataset(buffered_path)
        print(
            f"Area buffered by {buffer_size} pixel(s) already exists at {buffered_path}."
        )

    return buffered_ds, buffered_path


def find_chunks_with_percentage(
    array: np.ndarray,
    range_min: float,
    range_max: float,
    chunk_size: int,
    threshold_percent: float,
) -> list[tuple[float, float]]:
    """Find chunks in the array that contain a certain threshold percentage of pixel values within a specified range.

    Parameters
        array (ndarray): Input array.
        range_min (float): Minimum value for the range.
        range_max (float): Maximum value for the range.
        chunk_size (int): Size of the chunks in rows and columns.
        threshold_percent (float): Threshold percentage for chunk selection.

    Returns:
        tuple[list[tuple[float, float]], list[float]]: List of chunk coordinates that meet the threshold percentage
            criteria and list of perceentage of gridcell meeting criteria.
    """
    # to make chunk_size behave as expected in the face of non-inclusive final indices
    chunk_size = chunk_size + 1
    rows, cols = array.shape
    chunk_rows = np.arange(0, rows - chunk_size, chunk_size)
    chunk_cols = np.arange(0, cols - chunk_size, chunk_size)

    chunk_coords = []
    cell_coverages = []
    with tqdm(
        total=(len(chunk_rows)) * (len(chunk_cols)),
        desc="Calculating area within range",
    ) as pbar:
        for start_row in chunk_rows:
            for start_col in chunk_cols:
                # update tqdm progress bar
                pbar.update(1)
                # amount of cell covered by values within range as percentage
                cell_coverage = (
                    np.mean(
                        np.logical_and(
                            array[
                                start_row : start_row + chunk_size,  # noqa
                                start_col : start_col + chunk_size,  # noqa
                            ]
                            >= range_min,
                            array[
                                start_row : start_row + chunk_size,  # noqa
                                start_col : start_col + chunk_size,  # noqa
                            ]
                            <= range_max,
                        )
                    )
                    * 100
                )
                if cell_coverage >= threshold_percent:
                    chunk_coords.append(
                        (
                            (start_row, start_col),
                            (start_row + chunk_size - 1, start_col + chunk_size - 1),
                        )
                    )
                    cell_coverages.append(cell_coverage.item())
    return chunk_coords, cell_coverages


def index_to_coord(xa_da: xa.DataArray, index: tuple[int, int]) -> tuple[float, float]:
    """Convert a pair of indices of a DataArray to their corresponding coordinate values.

    Parameters
        xa_da (xa.DataArray): Input DataArray.
        index (tuple[int, int]): Index tuple in the form (row_index, col_index).

    Returns:
        tuple[float, float]: Tuple of latitude and longitude coordinate values corresponding to the index.

    """
    lon = xa_da.longitude.values[index[1]]
    lat = xa_da.latitude.values[index[0]]
    return lon, lat


def delta_index_to_distance(
    xa_da: xa.DataArray, start_index: tuple[int, int], end_index: tuple[int, int]
) -> tuple[float, float]:
    """Convert the delta index between two coordinates to their corresponding distance values.

    Parameters
        xa_da (xarray.DataArray): Input DataArray.
        start_index (tuple[int, int]): Start index tuple in the form (start_row_index, start_col_index).
        end_index (tuple[int, int]): End index tuple in the form (end_row_index, end_col_index).

    Returns:
        tuple[float, float]: Tuple of delta latitude and delta longitude distance values between the coordinates.
    """
    # determine start and end coordinates
    start_coords = index_to_coord(xa_da, start_index)
    end_coords = index_to_coord(xa_da, end_index)
    # calculate differences
    diffs = np.subtract(end_coords, start_coords)
    delta_y, delta_x = diffs[0], diffs[1]
    return delta_y, delta_x


def encode_nans_one_hot(array: np.ndarray, all_nan_dims: int = 1) -> np.ndarray:
    """One-hot encode NaN values in a 3D array.

    Parameters
    ----------
        array (np.ndarray) The input 3D array.
        all_nan_dims (int, optional): The number of dimensions (starting from the second dimension) to consider when
            determining if all values are NaN. Default is 1.

    Returns
    -------
        np.ndarray: The one-hot encoded array with NaN information.
    """
    # boolean mask of land (where all variable values are nan throughout all time)
    land_mask = np.all(np.isnan(array), (1, 2))
    # binary land mask
    onehot_column = np.where(land_mask, 1, 0)
    # binary land mask expanded to target dimensions
    onehot_expanded = np.expand_dims(onehot_column, axis=(1, 2))
    # binary land mask broadcast back through time
    onehot_broadcast = np.repeat(onehot_expanded, array.shape[1], axis=1)

    return np.concatenate((array, onehot_broadcast), axis=2)


def add_gt_to_xa_d(
    xa_d: xa.DataArray | xa.Dataset,
    gt_da: xa.DataArray,
    gt_name: str = "coral_algae_gt",
) -> xa.Dataset:
    """Add a ground truth data array to an xarray DataArray or Dataset.

    Parameters
    ----------
        xa_d (xarray.DataArray or xarray.Dataset): Input xarray DataArray or Dataset.
        gt_da (xarray.DataArray): Ground truth DataArray to be added.
        gt_name (str, optional): Name of the ground truth variable. Defaults to "coral_algae_gt".

    Returns
    -------
        xarray.Dataset: Updated xarray Dataset with the ground truth variable added.
    """
    gt_da = gt_da.isel(latitude=slice(None, None, -1))
    if "time" in gt_da.dims:
        gt_da = gt_da.isel(time=0)
    expanded_da = np.tile(gt_da, (len(list(xa_d.time.values)), 1, 1)).astype("int")

    xa_d[gt_name] = (("time", "latitude", "longitude"), expanded_da)
    return xa_d


def generate_and_add_gt_to_xa_d(
    xa_d: xa.DataArray | xa.Dataset,
    gt_da: xa.DataArray,
    lat_lon_starts=(-10, 141.95),
    coord_range=(-7.01, 5.11),
    gt_name: str = "coral_algae_gt",
) -> xa.Dataset:
    """Generate and add ground truth (gt) data to an xarray dataset or data array.

    Parameters
    ----------
        xa_d (xa.DataArray or xa.Dataset): Input xarray dataset or data array.
        gt_da (xa.DataArray): Ground truth data array to be added.
        lat_lon_starts (tuple[float, float], optional): Latitude and longitude starting values
            for generating ground truth data. Defaults to (-10, 141.95).
        coord_range (tuple[float, float], optional): Range of latitude and longitude coordinates
            for generating ground truth data. Defaults to (-7.01, 5.11).
        gt_name (str, optional): Name of the ground truth variable. Defaults to "coral_algae_gt".

    Returns
    -------
        xa.Dataset or xa.DataArray: Xarray dataset or data array with ground truth data added.
    """
    # if lat_lon_starts and coord_range activately not supplied (indicating considering whole dataset)
    if lat_lon_starts is None and coord_range is None:
        lat_lims = xarray_coord_limits(xa_d, "latitude")
        lon_lims = xarray_coord_limits(xa_d, "longitude")
        lat_lon_starts = (lat_lims[0], lat_lims[1]), (lon_lims[0], lon_lims[1])
        coord_range = (np.diff(lat_lims).item(), np.diff(lon_lims).item())

    gt, _ = sample_spatial_batch(
        gt_da, lat_lon_starts=lat_lon_starts, coord_range=coord_range
    )

    return add_gt_to_xa_d(xa_d, gt)


def ds_subsample_from_coord(
    xa_ds: xa.Dataset, chunk_coords: tuple[float, float]
) -> xa.Dataset:
    """Subsample an xarray Dataset based on given latitude and longitude chunk coordinates.

    Parameters
    ----------
    xa_ds (xa.Dataset): Input xarray Dataset.
    chunk_coords (tuple[float, float]): Tuple containing the latitude and longitude chunk coordinates.

    Returns
    -------
    xa.Dataset: Subsampled xarray Dataset.
    """
    lats, lons = index_pair_to_lats_lons_pair(chunk_coords)
    return xa_ds.isel(
        {"latitude": slice(lats[0], lats[1]), "longitude": slice(lons[0], lons[1])}
    )


def get_vars_from_ds_or_da(xa_d: xa.DataArray | xa.Dataset) -> str | list[str]:
    """Get the variable name(s) from an xarray Dataset or DataArray.

    Parameters
    ----------
    xa_d (xa.DataArray | xa.Dataset): Input xarray Dataset or DataArray.

    Returns
    -------
    str | list[str]: Variable name(s).
    """
    if xa_d.isinstance(xa.core.dataarray.DataArray):
        vars = xa_d.name
    elif xa_d.isinstance(xa.core.dataarray.Dataset):
        vars = list(xa_d.data_vars)
    else:
        raise TypeError("Format was neither an xarray Dataset nor a DataArray")

    return vars


def index_pair_to_lats_lons_pair(
    coord_pair: tuple[float],
) -> tuple[tuple[float], tuple[float]]:
    """Convert index pair to latitude and longitude pair.

    Parameters
    ----------
    coord_pair (tuple[float]): Index pair containing the start and end coordinates.

    Returns
    -------
    tuple[tuple[float], tuple[float]]: Latitude and longitude pair.
    """
    starts, ends = coord_pair[0], coord_pair[1]
    lats, lons = (starts[0], ends[0]), (starts[1], ends[1])

    return lats, lons


def calculate_coord_resolution(
    xa_d: xa.Dataset | xa.DataArray, coord: str
) -> tuple[float]:
    """Calculate the spatial resolution of latitude and longitude in an xarray Dataset or DataArray.

    Parameters
    ----------
    xa_d (xa.Dataset | xa.DataArray): Input xarray Dataset or DataArray.
    coord (str): Coordinate to calculate the resolution of.

    Returns
    -------
    tuple[float]: Spatial resolution of latitude and longitude.
    """
    return np.mean(np.diff(xa_d[coord].values))


def calculate_spatial_resolution(xa_d: xa.Dataset | xa.DataArray) -> tuple[float]:
    """Calculate the spatial resolution of latitude and longitude in an xarray Dataset or DataArray.

    Parameters
    ----------
    xa_d (xa.Dataset | xa.DataArray): Input xarray Dataset or DataArray.

    Returns
    -------
    tuple[float]: Spatial resolution of latitude and longitude.
    """
    # average latitudinal resolution
    lat_resolution = calculate_coord_resolution(xa_d, "latitude")
    # average longitudinal resolution
    lon_resolution = calculate_coord_resolution(xa_d, "longitude")

    return lat_resolution, lon_resolution


def nc_chunk_files(
    dest_dir_path: Path | str,
    xa_ds: xa.Dataset,
    chunk_size: int = 20,
    threshold_percent: float = 10,
    vmin: float = -100,
    vmax: float = 0,
):
    """Save chunks of an xarray Dataset to NetCDF files along with accompanying metadata JSON files.

    Parameters
    ----------
    dest_dir_path (Path | str): Directory path to save the chunk files.
    xa_ds (xa.Dataset): Input xarray Dataset.
    chunk_size (int, optional): Size of the chunks (default is 20).
    threshold_percent (float, optional): Threshold percentage for chunk coverage (default is 10).
    vmin (float, optional): Minimum value for chunk selection (default is -100).
    vmax (float, optional): Maximum value for chunk selection (default is 0).

    Returns
    -------
    None
    """

    chunk_coord_pairs, coverages = find_chunks_with_percentage(
        xa_ds, vmin, vmax, chunk_size, threshold_percent
    )

    for i, coord_pair in tqdm(
        enumerate(chunk_coord_pairs),
        desc="Saving chunks .nc and accompanying metadata .json files",
        total=len(chunk_coord_pairs),
    ):
        sub_ds = ds_subsample_from_coord(xa_ds, coord_pair)
        # generate filename and file_path
        filename = "_".join(
            ("chunk", utils.pad_number_with_zeros(number=i, resulting_len=3))
        )
        file_path = Path(dest_dir_path) / filename
        # generate chunk metadata
        info_dict = generate_chunk_json(
            sub_ds, file_path, coord_pair=coord_pair, coverage=coverages[i]
        )
        # save metadata file
        file_ops.save_json(
            info_dict, filepath=file_path.with_suffix(".json"), verbose=False
        )
        # save nc file
        sub_ds.to_netcdf(path=file_path.with_suffix(".nc"))

    print(f".nc chunk files and accompanying metadata written to {str(dest_dir_path)}")
    return chunk_coord_pairs, coverages


def generate_chunk_json(
    xa_d: xa.DataArray,
    file_path: str | Path,
    coord_pair: tuple[int] = None,
    coverage: float = None,
) -> dict:
    """Generate metadata JSON dictionary for a chunk of an xarray DataArray.

    Parameters
    ----------
    xa_d (xa.DataArray): Input xarray DataArray.
    file_path (str | Path): File path of the chunk.
    coord_pair (tuple[int]): Index pair containing the start and end coordinates.
    coverage (float): Chunk coverage value.

    Returns
    -------
    dict: Metadata JSON dictionary.
    """

    # TODO: make robust with different dataset/dataarray

    # make filename
    vars = get_vars_from_ds_or_da(xa_d)
    # convert coord indices to absolute coords (TODO: could add spatial functionality for chunking)
    lat_lims = xarray_coord_limits(xa_d, "latitude")
    lon_lims = xarray_coord_limits(xa_d, "longitude")
    # find resolution
    lat_resolution_d, lon_resolution_d = calculate_spatial_resolution(xa_d)
    lat_resolution_m, lon_resolution_m = degrees_to_distances(
        lat_resolution_d, lon_resolution_d
    )
    # calculate minimum and maximum bathymetries
    min_bath, max_bath = xa_d.values.min(), xa_d.values.max()

    info_dict = {
        "file name": file_path.stem,
        "file path": str(file_path),
        "variables": vars,
        "latitude range": lat_lims,
        "longitude range": lon_lims,
        "latitude resolution (degrees)": lat_resolution_d,
        "longitude resolution (degrees)": lon_resolution_d,
        "latitude resolution (meters)": lat_resolution_m,
        "longitude resolution (meters)": lon_resolution_m,
        "minimum bathymetry": min_bath,
        "maximum bathymetry": max_bath,
    }

    if coord_pair is not None:
        additional_info = {
            "latitude chunk size": np.diff((coord_pair[0][0], coord_pair[1][0])).item(),
            "longitude chunk size": np.diff(
                (coord_pair[0][1], coord_pair[1][1])
            ).item(),
            "start index pair": coord_pair[0],
            "end index pair": coord_pair[1],
        }
        info_dict = info_dict.update(additional_info)
    if coverage is not None:
        additional_info = {
            "cell coverage": coverage,
        }
        info_dict = info_dict.update(additional_info)

    return info_dict


def find_coord_indices(
    xa_da: xa.Dataset | xa.DataArray, lat: float, lon: float
) -> tuple[int, int]:
    # Get the latitude and longitude spacing

    lat_values = list(xa_da.latitude.values)
    lon_values = list(xa_da.longitude.values)

    lat_index = np.where(np.isclose(lat_values, lat))[0][0]
    lon_index = np.where(np.isclose(lon_values, lon))[0][0]

    return lat_index, lon_index


def drop_xa_variables(
    xa_d: xa.DataArray | xa.Dataset,
    dim_drops: list[str] = ["time"],
    var_drops: list[str] = ["spatial_ref", "depth", "band"],
) -> xa.Dataset:
    # Drop dimensions if they exist in the Dataset
    for dim in dim_drops:
        if dim in xa_d.dims:
            xa_d = xa_d.drop_dims(dim)

    # Drop variables if they exist in the DataArray
    for var in var_drops:
        if isinstance(xa_d, xa.DataArray):
            raise TypeError(
                f"Object is a DataArray and therefore cannot drop variables {var_drops}."
            )
        else:
            if var in xa_d.variables:
                xa_d = xa_d.drop_vars(var)
    return xa_d


def generate_var_mask(
    xa_d: xa.Dataset | xa.DataArray,
    mask_var: str = "elevation",
    limits: tuple[float] = [-2000, 0],
    comp_mask_var: str = "unep_coral_presence",
) -> xa.DataArray:
    if isinstance(xa_d, xa.DataArray):
        return (xa_d <= max(limits)) & (xa_d >= min(limits))

    elif isinstance(xa_d, xa.Dataset):
        if mask_var not in xa_d.variables:
            raise ValueError(f"Variable {mask_var} not found in dataset.")
        else:
            return (xa_d[mask_var] <= max(limits)) & (xa_d[mask_var] >= min(limits))

    else:
        raise TypeError(
            f"xa_d was neither an xarray Dataset nor a DataArray. Instead type: {type(xa_d)}."
        )


# def resample_list_xa_ds_into_dict(
#     xa_das: list[xa.DataArray],
#     target_resolution: float,
#     unit: str = "m",
#     lat_lims: tuple[float] = (-10, -17),
#     lon_lims: tuple[float] = (142, 147),
# ) -> dict:
#     """
#     Resample a list of xarray DataArrays to the target resolution and merge them.

#     Parameters
#     ----------
#         xa_das (list[xa.DataArray]): A list of xarray DataArrays to be resampled and merged.
#         target_resolution (float): The target resolution for resampling.
#         unit (str, defaults to "m"): The unit of the target resolution.
#         interp_method: (str, defaults to "linear") The interpolation method for resampling.

#     Returns
#     -------
#         A dictionary containing the resampled xarray DataArrays merged by their names.
#     """
#     # TODO: will probably need to save to individual files/folders and combine at test/train time
#     # may need to go to target array here
#     target_resolution_d = choose_resolution(target_resolution, unit)[1]

#     # dummy_xa = generate_dummy_xa(target_resolution_d, lat_lims, lon_lims)

#     resampled_xa_das_dict = {}
#     for xa_da in tqdm(xa_das, desc="Resampling xarray DataArrays"):
#         # xa_resampled = resample_xa_d_to_other(xa_da, dummy_xa, name=xa_da.name)
#         xa_resampled = resample_xarray_to_target(xa_da, target_resolution_d)
#         resampled_xa_das_dict[xa_da.name] = xa_resampled

#     return resampled_xa_das_dict


def combine_ds_tiles(
    xa_d_list: list[xa.DataArray | xa.Dataset], resolution_d: float
) -> xa.DataArray | xa.Dataset:
    """
    Combine a list of datasets using the combine_first() function.

    Parameters
    ----------
    datasets : list of xarray.Dataset
        List of datasets to be combined.

    Returns
    -------
    xarray.Dataset
        Combined dataset.
    """
    if len(xa_d_list) == 1:
        print("Single object provided. No combination necessary")
        return xa_d_list[0]

    # initialise a dummy dataarray across total region to combine to allow reprojection
    (lat_lims), (lon_lims) = min_max_of_all_spatial_coords(xa_d_list)
    dummy_xa = generate_dummy_xa(resolution_d, lat_lims, lon_lims)

    processed_list = []
    for i, ds in tqdm(
        enumerate(xa_d_list),
        total=len(xa_d_list[1:]),
        desc="Pre-processing xarray objects",
    ):
        ds = ds.rename({"latitude": "y", "longitude": "x"})
        processed_list.append(ds.rio.reproject_match(dummy_xa))

    # Initialize the combined dataset with the first dataset in the list
    combined_ds = processed_list[0]

    # Iterate over the remaining datasets and combine them with the combined dataset
    for ds in tqdm(
        processed_list[1:], total=len(processed_list[1:]), desc=" Combining data"
    ):
        combined_ds = combined_ds.combine_first(ds)

    return process_xa_d(combined_ds)

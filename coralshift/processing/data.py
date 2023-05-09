import xarray as xa
import rioxarray as rio
import numpy as np
import rasterio
import pandas as pd
import haversine

from rasterio import features
from tqdm import tqdm
from scipy.ndimage import binary_dilation
from pathlib import Path
from coralshift.utils import file_ops


def upsample_xarray(xa_array: xa.DataArray, factors: dict) -> xa.DataArray:
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


def process_xa_array(
    xa_array: xa.DataArray,
    coords_to_drop: list[str],
    coords_to_rename: dict = {"x": "longitude", "y": "latitude"},
    resolution: float = None,
    shape: tuple[int] = None,
    verbose: bool = True,
) -> xa.DataArray:
    """Process the given xarray DataArray by dropping and renaming specified coordinates.

    Parameters
    ----------
        xa_array (xa.DataArray): xarray DataArray to be processed.
        coords_to_drop (list[str]): list of coordinates to be dropped from the DataArray.
        coords_to_rename (dict, optional): dictionary of coordinates to be renamed in the DataArray.
            Defaults to {"x": "longitude", "y": "latitude"}.
        resolution (float, optional): Output resolution of the upsampled DataArray, in the same units as the input.
            Defaults to None.
        shape (tuple, optional): Shape of the output DataArray as (height, width). If specified, overrides the
            resolution parameter. Defaults to None.
        verbose (bool, optional): if True, print information about the remaining coordinates in the DataArray.
            Defaults to True.

    Returns
    -------
        xa.DataArray: The processed xarray DataArray.
    """
    # drop specified coordinates
    xa_array = xa_array.drop_vars(coords_to_drop)
    # rename specified coordinates
    xa_array = xa_array.rename(coords_to_rename)

    if resolution or shape:
        upsample_xa_array(xa_array, resolution=resolution, shape=shape)
    if verbose:
        # show info about remaining coords
        print(xa_array.coords)

    return xa_array


def process_xa_arrays_in_dict(
    xa_array_dict: dict,
    coords_to_drop: list[str],
    coords_to_rename: dict = {"x": "longitude", "y": "latitude"},
    resolution: float = None,
    shape: tuple[int] = None,
) -> dict:
    """Process multiple xarray DataArrays stored in a dictionary by dropping and renaming specified coordinates.

    Parameters
    ----------
        xa_array_dict (dict): A dictionary of xarray DataArrays to be processed: keys = filename, values = xa.DataArray.
        coords_to_drop (list[str]): A list of coordinate fields to be dropped from each DataArray.
        coords_to_rename (dict): A dictionary of coordinate fields to be renamed in each DataArray.

    Returns
    -------
        dict: A dictionary containing the processed xarray DataArrays.
    """
    processed_dict = {}
    print(
        f"Processing xa_arrays in dictionary. Dropping {coords_to_drop}, renaming {coords_to_rename.keys()}."
    )
    for name, xa_array in tqdm(xa_array_dict.items()):
        processed_dict[name] = process_xa_array(
            xa_array, coords_to_drop, coords_to_rename, resolution, shape, verbose=False
        )

    return processed_dict


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
        filename = str(file_ops.get_n_last_subparts_path(tif, 1))
        tif_array = rio.open_rasterio(tif)
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


def min_max_of_coords(xa_array: xa.DataArray, coord: str) -> tuple[float, float]:
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


def return_pixels_closest_to_value(
    array: np.ndarray,
    central_value: float,
    tolerance: float = 0.5,
    buffer_pixels: int = 10,
    bathymetry_only: bool = True,
) -> np.ndarray:
    """Returns a 1D array of all the pixels in the input array that are closest to a specified central value within a
    given tolerance and within a pixel buffer zone.

    Parameters
    ----------
    array (np.ndarray): The input array of pixel values.
    central_value (float): The central value to which the pixels should be compared.
    tolerance (float, optional): The tolerance within which the pixels are considered to be "close" to the central
        value. Defaults to 0.5.
    buffer_pixels (int, optional): The size of the buffer zone around the pixels. Defaults to 10.
    bathymetry_only (bool, optional): Whether to only consider bathymetric data, i.e., values less than zero.
        Defaults to True.

    Returns
    -------
    np.ndarray: A 1D array of all the pixels in the input array that are closest to the specified central value within
        the given tolerance and within the pixel buffer zone.
    """
    binary = np.isclose(array, central_value, atol=0.5)
    # morphological dilation operation
    dilated = binary_dilation(binary, iterations=buffer_pixels)

    array_vals = array[dilated]
    # if specifying only bathymetric data
    if bathymetry_only:
        array_vals = array_vals[array_vals < 0]

    # return only non-zero values as 1d array
    return array_vals[np.nonzero(array_vals)]


def return_distance_closest_to_value(
    array: np.ndarray,
    central_value: float,
    tolerance: float = 0.5,
    buffer_distance: float = 300,
    distance_per_pixel: float = 30,
    bathymetry_only: bool = True,
) -> np.ndarray:
    """Wrapper for return_pixels_closest_to_value() allowing specification by distance from thresholded values rather
    than number of pixels

    Returns a 1D array of all the pixels in the input array that are closest to a specified central value within a
    given tolerance and within a distance buffer zone.

       Parameters
    ----------
    array (np.ndarray): The input array of pixel values.
    central_value (float): The central value to which the pixels should be compared.
    tolerance (float, optional): The tolerance within which the pixels are considered to be "close" to the central
        value. Defaults to 0.5.
    buffer_distance (float, optional): The size of the buffer zone around the pixels. Defaults to 300.
    bathymetry_only (bool, optional): Whether to only consider bathymetric data, i.e., values less than zero.
        Defaults to True.

    Returns
    -------
    np.ndarray: A 1D array of all the pixels in the input array that are closest to the specified central value within
        the given tolerance and within the distance buffer zone.
    """
    buffer_pixels = buffer_distance / distance_per_pixel
    return return_pixels_closest_to_value(
        array, central_value, tolerance, buffer_pixels, bathymetry_only
    )


def upsample_xa_array(
    xa_array: xa.DataArray, resolution: float = 1 / 12, shape: tuple = None
) -> xa.DataArray:
    """Upsamples the resolution of a DataArray using rioxarray's 'reproject' functionality: reprojecting it onto a lower
    resolution and/or differently-sized grid

    Parameters
    ----------
    xa_array (xa.DataArray): Input DataArray to upsample.
    resolution (float, optional): Output resolution of the upsampled DataArray, in the same units as the input.
        Defaults to 1/12.
    shape (tuple, optional): Shape of the output DataArray as (height, width). If specified, overrides the resolution
        parameter.

    Returns
    -------
    xa.DataArray: The upsampled DataArray
    """

    if not shape:
        upsampled_array = xa_array.rio.reproject(
            xa_array.rio.crs, resolution=resolution
        )
    else:
        upsampled_array = xa_array.rio.reproject(xa_array.rio.crs, shape=shape)

    return upsampled_array


def upsample_dict_of_xa_arrays(
    xa_dict: dict, resolution: float = 1 / 12, shape: tuple[int, int] = None
) -> dict:
    """Upsamples the resolution of each DataArray in a dictionary and returns the upsampled dictionary.

    Parameters
    ----------
    xa_dict (dict): Dictionary containing the input DataArrays.
    resolution (float, optional): Output resolution of the upsampled DataArrays, in the same units as the input.
        Defaults to 1/12.
    shape (tuple, optional): Shape of the output DataArrays as (height, width). If specified, overrides the resolution
        parameter.

    Returns
    -------
    dict: Dictionary containing the upsampled DataArrays with corresponding keys as array_name_upsampled names.
    TODO: add in check to ensure that upsampling rather than downsampling?
    """
    upsampled_dict = {}
    print(f"Upsampling {xa_dict.keys()} from dictionary to {resolution}{shape}.")
    for name, array in tqdm(xa_dict.items()):
        upsampled_name = "_".join(
            (file_ops.remove_suffix(name), "upsampled", "{0:.3g}".format(resolution))
        )
        upsampled_array = upsample_xa_array(array, resolution, shape)
        upsampled_dict[upsampled_name] = upsampled_array

    return upsampled_dict


def xarray_coord_limits(xa_array: xa.Dataset, dim: str) -> tuple[float]:
    """Compute the minimum and maximum values for a coordinate dimension of an xarray dataset.

    Parameters
    ----------
        xa_array (xa.Dataset): input xarray dataset.
        dim (str): coordinate name.

    Returns:
        tuple[float]: minimum and maximum values of the coordinate

    """
    min, max = float(xa_array[dim].min()), float(xa_array[dim].max())
    return (min, max)


def dict_xarray_coord_limits(xa_array: xa.Dataset) -> dict:
    """Compute the minimum and maximum values for each coordinate dimension of an xarray dataset and assign to a dict

    Parameters
    ----------
        xa_array (xa.Dataset): input xarray dataset.

    Returns:
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
    np.ndarray: A numpy array with the rasterized data.
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
        rasterized = features.rasterize(
            shapes=shapes,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            all_touched=True,
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
    raster (np.ndarray): A numpy array with the raster data.
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
):
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
        filename.
        Either way, returns raster
    """
    filepath = file_ops.add_suffix_if_necessary(Path(dir_path) / filename, ".nc")
    # if specifying resolution as horizontal distance rather than degrees lat/lon, calculate approximate degrees
    if horizontal_distance:
        resolution = distance_to_degree(horizontal_distance)

    if not file_ops.check_file_exists(dir_path, filename, ".nc"):
        raster_xa = generate_raster_xa(
            xa_ds, resolution, class_col, xa_name, all_touched, filepath
        )
        return raster_xa
    else:
        print(f"{filename} already exists in {dir_path}. No files written.")
        return xa.open_dataset(filepath)


def distance_to_degree(
    distance: float, approx_lat: float = -18, approx_lon: float = 145
) -> float:
    """Converts a distance in meters to the corresponding distance in degrees, given an approximate location on Earth.

    Parameters
    ----------
    distance (float): The distance in meters.
    approx_lat (float, optional): The approximate latitude of the location in degrees. Defaults to -18.0.
    approx_lon (float, optional): The approximate longitude of the location in degrees. Defaults to 145.0.

    Returns
    -------
    float: The corresponding distance in degrees.
    """
    # calculate the coordinates 'distance' meters to the southwest (chosen to give measure of both lat and lon)
    (lat_deg, lon_deg) = haversine.inverse_haversine(
        (approx_lat, approx_lon),
        distance,
        haversine.Direction.SOUTHWEST,
        unit=haversine.Unit.METERS,
    )
    delta_lat, delta_lon = abs(lat_deg - approx_lat), abs(lon_deg - approx_lon)
    # return hypotenuse (encapsulates difference in both lat and lon)
    return np.hypot(delta_lat, delta_lon)


def filter_strings(
    str_list: list[str], exclude: list[str] = ["latitude", "longitude", "depth", "time"]
):
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
    for var in tqdm(variables_to_read):
        vals = ds_stacked[var].values
        array_list.append(vals)

    # move location to first column. New shape: grid_cell_val x var x time
    return np.moveaxis(np.array(array_list), 2, 0)

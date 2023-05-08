import xarray as xa
import rioxarray as rio
import numpy as np

# import rasterio as rio
from tqdm import tqdm
from scipy.ndimage import binary_dilation

# import numpy as np
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
    verbose: bool = True,
) -> xa.DataArray:
    """Process the given xarray DataArray by dropping and renaming specified coordinates.

    Parameters
    ----------
        xa_array (xa.DataArray): xarray DataArray to be processed.
        coords_to_drop (list[str]): list of coordinates to be dropped from the DataArray.
        coords_to_rename (dict, optional): dictionary of coordinates to be renamed in the DataArray.
            Defaults to {"x": "longitude", "y": "latitude"}.
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

    if verbose:
        # show info about remaining coords
        print(xa_array.coords)

    return xa_array


def process_xa_arrays_in_dict(
    xa_array_dict: dict,
    coords_to_drop: list[str],
    coords_to_rename: dict = {"x": "longitude", "y": "latitude"},
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
    for name, xa_array in xa_array_dict.items():
        processed_dict[name] = process_xa_array(
            xa_array, coords_to_drop, coords_to_rename, verbose=False
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
        xa_array_dict[filename] = tif_array

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


def reduce_xa_array(
    xa_array: xa.DataArray, resolution: float = 0.01, shape: tuple = None
) -> xa.DataArray:
    """Reduces the resolution of a DataArray using rioxarray's 'reproject' functionality: reprojecting it onto a lower
    resolution and/or differently-sized grid

    Parameters
    ----------
    xa_array (xa.DataArray): Input DataArray to reduce.
    resolution (float, optional): Output resolution of the reduced DataArray, in the same units as the input.
        Defaults to 0.01.
    shape (tuple, optional): Shape of the output DataArray as (height, width). If specified, overrides the resolution
        parameter.

    Returns
    -------
    xa.DataArray: The reduced DataArray
    """

    if not shape:
        reduced_array = xa_array.rio.reproject(xa_array.rio.crs, resolution=resolution)
    else:
        reduced_array = xa_array.rio.reproject(xa_array.rio.crs, shape=shape)

    return reduced_array


def reduce_dict_of_xa_arrays(
    xa_dict: dict, resolution: float = 0.01, shape: tuple = None
) -> dict:
    """Reduces the resolution of each DataArray in a dictionary and returns the reduced dictionary.

    Parameters
    ----------
    xa_dict (dict): Dictionary containing the input DataArrays.
    resolution (float, optional): Output resolution of the reduced DataArrays, in the same units as the input. Defaults
        to 0.01.
    shape (tuple, optional): Shape of the output DataArrays as (height, width). If specified, overrides the resolution
        parameter.

    Returns
    -------
    dict: Dictionary containing the reduced DataArrays with corresponding keys as array_name_reduced
    names.
    """
    reduced_dict = {}
    for name, array in tqdm(xa_dict.items()):
        reduced_name = file_ops.remove_suffix(name) + "_reduced"
        reduced_array = reduce_xa_array(array, resolution, shape)
        reduced_dict[reduced_name] = reduced_array

    return reduced_dict


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

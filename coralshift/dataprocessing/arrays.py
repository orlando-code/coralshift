import xarray as xa
import numpy as np

# import rasterio as rio
from coralshift import file_ops
from tqdm import tqdm
from scipy.ndimage import binary_dilation


def reduce_xa_array(
    array: xa.DataArray, resolution: float = 0.01, shape: tuple = None
) -> xa.DataArray:
    """Reduce the resolution of an xarray DataArray by reprojecting it onto a lower resolution and/or differently-sized
    grid

    Parameters
    ----------
        array (xa.DataArray): input DataArray.
        resolution (float): new resolution of the output DataArray, in the units of the input DataArray's CRS.
            Default is 0.01.
        shape (tuple): new shape of the output DataArray, if resolution not specified. Specified as a tuple of
            (height, width).

    Returns
    -------
        xa.DataArray: The output DataArray, at lower resolution or a new shape.
    """
    if not shape:
        reduced_array = array.rio.reproject(array.rio.crs, resolution)
    else:
        reduced_array = array.rio.reproject(array.rio.crs, shape=shape)

    return reduced_array


def reduce_dict_of_xa_arrays(
    xa_dict: dict, resolution: float = 0.01, shape: tuple = None
) -> dict:
    """Reduce the resolution for each array in a a dictionary of xarray DataArrays.

    Parameters
    ----------
        xa_dict (xa.DataArray): dictionary of xa.DataArrays. Keys are array names, values are arrays.
        resolution (float): new resolution of the output DataArray, in the units of the input DataArray's CRS.
            Default is 0.01.
        shape (tuple): new shape of the output DataArray, if resolution not specified. Specified as a tuple of
            (height, width).

    Returns
    -------
        xa.DataArray: dictionary with keys as array_name_reduced, values reduced arrays.
    """
    reduced_dict = {}
    for name, array in tqdm(xa_dict.items()):
        reduced_name = file_ops.remove_suffix(name) + "_reduced"
        reduced_array = reduce_xa_array(array, resolution, shape)

        reduced_dict[reduced_name] = reduced_array

    return reduced_dict


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

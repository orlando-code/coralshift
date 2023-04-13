import xarray as xa
import rioxarray as rio
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
    xa_array: xa.DataArray, coords_to_drop: list[str], coords_to_rename: dict = {"x": "longitude", "y": "latitude"},
        verbose: bool = True) -> xa.DataArray:
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
    xa_array_dict: dict, coords_to_drop: list[str],
        coords_to_rename: dict = {"x": "longitude", "y": "latitude"}) -> dict:
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
        processed_dict[name] = process_xa_array(xa_array, coords_to_drop, coords_to_rename, verbose=False)

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

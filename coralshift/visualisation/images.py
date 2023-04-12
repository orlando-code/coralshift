import xarray as xa
from coralshift.processing import data


def display_xa_array(xa_array: xa.DataArray, upsampling: dict = {"x": 1000, "y": 1000}) -> xa.DataArray:
    """Displays an xarray DataArray as a plot, optionally with upsampling to increase the resolution.

    Parameters
    ----------
    xa_array (xr.DataArray): The xarray DataArray to be displayed.
    upsampling (dict, optional): A dictionary specifying the upsampling factor for each dimension. The keys should be
        the names of the dimensions, and the values should be integers indicating the factor by which to increase the
        resolution. Default is {"x": 1000, "y": 1000}.

    Returns
    -------
    xr.DataArray: The xarray DataArray that was displayed.
    """
    min_val, max_val = xa_array.values.min(), xa_array.values.max()

    if upsampling:
        xa_array = data.upsample_xarray(xa_array, upsampling)

    xa_array.plot(cmap='gist_earth', vmin=min_val, vmax=max_val)
    return xa_array

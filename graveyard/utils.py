# general
import numpy as np
import pandas as pd
from datetime import datetime

# spatial
import geopandas as gpd

# file handling
# from pathlib import Path

# spatial
import xarray as xa

# custom
from coralshift.processing import spatial_data


def get_resolution(xa_d: xa.Dataset | xa.DataArray) -> float:
    """
    Get the resolution of an xarray dataset or data array.

    Parameters:
    - xa_d: The xarray dataset or data array.

    Returns:
    - The resolution of the xarray dataset or data array.
    """
    if "latitude" in xa_d.coords:
        lat_diff = xa_d.latitude.diff(dim="latitude").values[0]
    elif "lat" in xa_d.coords:
        lat_diff = xa_d.lat.diff(dim="lat").values[0]
    else:
        raise ValueError(
            "Latitude coordinate not found in the xarray dataset or data array."
        )

    if "longitude" in xa_d.coords:
        lon_diff = xa_d.longitude.diff(dim="longitude").values[0]
    elif "lon" in xa_d.coords:
        lon_diff = xa_d.lon.diff(dim="lon").values[0]
    else:
        raise ValueError(
            "Longitude coordinate not found in the xarray dataset or data array."
        )

    resolution = min(abs(lat_diff), abs(lon_diff))
    return resolution


def lat_lon_vals_from_geo_df(geo_df: gpd.geodataframe, resolution: float = 1.0):
    # Calculate the extent in degrees from bounds of geometry objects
    lon_min, lat_min, lon_max, lat_max = geo_df["geometry"].total_bounds
    # Calculate the width and height of the raster in pixels based on the extent and resolution
    width = int((lon_max - lon_min) / resolution)
    height = int((lat_max - lat_min) / resolution)

    return lon_min, lat_min, lon_max, lat_max, width, height


def calc_non_zero_ratio(df, predictand=None):
    # if df a series
    if isinstance(df, pd.Series):
        return np.where(df > 0, 1, 0).sum() / len(df)
    else:
        return np.where(df[predictand] > 0, 1, 0).sum() / len(df)


def lat_lon_string_from_tuples(
    lats: tuple[float, float], lons: tuple[float, float], dp: int = 0
):
    round_lats = iterative_to_string_list(lats, dp)
    round_lons = iterative_to_string_list(lons, dp)

    return (
        f"n{max(round_lats)}_s{min(round_lats)}_w{min(round_lons)}_e{max(round_lons)}"
    )


def iterative_to_string_list(iter_obj: tuple, dp: int = 0):
    # Round the values in the iterable object to the specified number of decimal places
    return [round(i, dp) for i in iter_obj]


def gen_seafloor_indices(xa_da: xa.Dataset, var: str, dim: str = "lev"):
    """Generate indices of seafloor values for a given variable in an xarray dataset.

    Args:
        xa_da (xa.Dataset): xarray dataset containing variable of interest
        var (str): name of variable of interest
        dim (str, optional): dimension along which to search for seafloor values. Defaults to "lev".

    Returns:
        indices_array (np.ndarray): array of indices of seafloor values for given variable
    """
    nans = np.isnan(xa_da[var]).sum(dim=dim)  # separate out
    indices_array = -(nans.values) - 1
    indices_array[indices_array == -(len(xa_da[dim].values) + 1)] = -1
    return indices_array


def extract_seafloor_vals(xa_da, indices_array):
    vals_array = xa_da.values
    t, j, i = indices_array.shape
    # create open grid for indices along each dimension
    t_grid, j_grid, i_grid = np.ogrid[:t, :j, :i]
    # select values from vals_array using indices_array
    return vals_array[t_grid, indices_array, j_grid, i_grid]


class FileName:
    def __init__(
        self,
        variable_id: str | list,
        grid_type: str,
        fname_type: str,
        date_range: str = None,
        lats: list[float, float] = None,
        lons: list[float, float] = None,
        levs: list[int, int] = None,
        plevels: list[float, float] = None,
    ):
        """
        Args:
            source_id (str): name of climate model
            member_id (str): model run
            variable_id (str): variable name
            grid_type (str): type of grid (tripolar or latlon)
            fname_type (str): type of file (individual, time-concatted, var_concatted) TODO
            lats (list[float, float], optional): latitude range. Defaults to None.
            lons (list[float, float], optional): longitude range. Defaults to None.
            plevels (list[float, float], optional): pressure level range. Defaults to None.
            fname (str, optional): filename. Defaults to None to allow construction.
        """
        self.variable_id = variable_id
        self.grid_type = grid_type
        self.fname_type = fname_type
        self.date_range = date_range
        self.lats = lats
        self.lons = lons
        self.levs = levs
        self.plevels = plevels

    def get_spatial(self):
        if self.lats and self.lons:  # if spatial range specified (i.e. cropping)
            # cast self.lats and self.lons lists to integers. A little crude, but avoids decimals in filenames
            lats = [int(lat) for lat in self.lats]
            lons = [int(lon) for lon in self.lons]
            return lat_lon_string_from_tuples(lats, lons).upper()
        else:
            return "uncropped"

    def get_plevels(self):
        if self.plevels == [-1] or self.plevels == -1:  # seafloor
            return f"sfl-{max(self.levs)}"
        elif not self.plevels:
            return "sfc"
            if self.plevels[0] is None:
                return "sfc"
        elif isinstance(self.plevels, float):  # specified pressure level
            return "{:.0f}".format(self.plevels / 100)
        elif isinstance(self.plevels, list):  # pressure level range
            if self.plevels[0] is None:  # if plevels is list of None, surface
                return "sfc"
            return f"levs_{min(self.plevels)}-{max(self.plevels)}"
        else:
            raise ValueError(
                f"plevel must be one of [-1, float, list]. Instead received '{self.plevels}'"
            )

    def get_var_str(self):
        if isinstance(self.variable_id, list):
            return "_".join(self.variable_id)
        else:
            return self.variable_id

    def get_grid_type(self):
        if self.grid_type == "tripolar":
            return "tp"
        elif self.grid_type == "latlon":
            return "ll"
        else:
            raise ValueError(
                f"grid_type must be 'tripolar' or 'latlon'. Instead received '{self.grid_type}'"
            )

    def get_date_range(self):
        if not self.date_range:
            return None
        if self.fname_type == "time_concatted" or self.fname_type == "var_concatted":
            # Can't figure out how it's being done currently
            return str("-".join((str(self.date_range[0]), str(self.date_range[1]))))
        else:
            return self.date_range

    def join_as_necessary(self):
        var_str = self.get_var_str()
        spatial = self.get_spatial()
        plevels = self.get_plevels()
        grid_type = self.get_grid_type()
        date_range = self.get_date_range()

        # join these variables separated by '_', so long as the variable is not None
        return "_".join(
            [i for i in [var_str, spatial, plevels, grid_type, date_range] if i]
        )

    def construct_fname(self):

        if self.fname_type == "var_concatted":
            if not isinstance(self.variable_id, list):
                raise TypeError(
                    f"Concatted variable requires multiple variable_ids. Instead received '{self.variable_id}'"
                )

        self.fname = self.join_as_necessary()

        return f"{self.fname}.nc"


def year_to_datetime(year: int, xa_d: xa.DataArray | xa.Dataset = None) -> datetime:
    """
    Convert an integer denoting a year to a datetime object.

    Args:
        year (int): The year to convert.
        xa_d (xa.DataArray | xa.Dataset, optional): An xarray DataArray or Dataset to check the year against.
        Defaults to None.

    Returns:
        datetime: The datetime object corresponding to the year.
    """
    # TODO: replace with datetime module
    if xa_d:
        # Extract the minimum and maximum years from the dataset's time coordinate
        min_year, max_year = spatial_data.min_max_of_coords(xa_d, "time")

        # Ensure the provided year is within the range of the dataset's time coordinate
        if year < min_year or year > max_year:
            raise ValueError(f"Year {year} is outside the dataset's time range")

    return datetime(year, 1, 1)

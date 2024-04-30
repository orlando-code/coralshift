from pathlib import Path
import re

import yaml
import sys
import concurrent.futures


import xarray as xa

from coralshift.cmipper import utils as cmipper_utils
from coralshift.utils import config


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
            return cmipper_utils.lat_lon_string_from_tuples(lats, lons).upper()
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


def find_files_for_time(filepaths, year_range):
    result = []
    for fp in filepaths:
        start_year = int(str(fp).split("_")[-1].split("-")[0][:4])
        end_year = (
            int(str(fp).split("_")[-1].split("-")[1][:4]) + 1
        )  # +1: XXXX12 = full XXXX year
        if (min(year_range) >= start_year) and (max(year_range)) <= end_year:
            result.append(fp)
    return result


def find_files_for_area(filepaths, lat_range, lon_range):
    result = []

    for filepath in filepaths:
        # uncropped refers to global coverage
        if "uncropped" in str(filepath):
            result.append(filepath)
            continue

        fp_lats, fp_lons = extract_lat_lon_ranges_from_fp(filepath)
        if (
            max(lat_range) <= max(fp_lats)
            and min(lat_range) >= min(fp_lats)
            and max(lon_range) <= max(fp_lons)
            and min(lon_range) >= min(fp_lons)
        ):
            result.append(filepath)

    return result


def extract_lat_lon_ranges_from_fp(file_path):
    # Define the regular expression pattern
    pattern = re.compile(
        r".*_n(?P<north>[\d.-]+)_s(?P<south>[\d.-]+)_w(?P<west>[\d.-]+)_e(?P<east>[\d.-]+).*.nc",
        re.IGNORECASE,
    )

    # Match the pattern in the filename
    match = pattern.match(str(Path(file_path.name)))

    if match:
        # Extract latitude and longitude values
        north = float(match.group("north"))
        south = float(match.group("south"))
        west = float(match.group("west"))
        east = float(match.group("east"))

        # Create lists of latitudes and longitudes
        lats = [north, south]
        lons = [west, east]
        return lats, lons
    else:
        return [-9999, -9999], [-9999, -9999]


def read_yaml(yaml_path: str | Path):
    with open(yaml_path, "r") as file:
        yaml_info = yaml.safe_load(file)
    return yaml_info


def redirect_stdout_stderr_to_file(filename):
    sys.stdout = open(filename, "w")
    sys.stderr = sys.stdout


def reset_stdout_stderr():
    """TODO: doesn't reset when in Jupyter Notebooks, works fine between files"""
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    if sys.stdout is not sys.__stdout__:
        sys.stdout.close()  # Close the file handle opened for stdout
        sys.stdout = sys.__stdout__
    if sys.stderr is not sys.__stderr__:
        sys.stderr.close()  # Close the file handle opened for stderr
        sys.stderr = sys.__stderr__


def execute_functions_in_threadpool(func, args):
    # hardcoding mac_workers so as to not cause issues on sherwood
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        futures = [executor.submit(func, *arg) for arg in args]
        return futures


def handle_errors(futures):
    for future in futures:
        try:
            future.result()
        except Exception as e:
            print(f"An error occurred: {e}")


# NEEDS WORK
################################################################################


def find_intersecting_cmip(
    variables: list[str],
    source_id: str = "EC-Earth3P-HR",
    member_id: str = "r1i1p2f1",
    lats: list[float, float] = [-40, 0],
    lons: list[float, float] = [130, 170],
    year_range: list[int, int] = [1950, 2014],
    levs: list[int, int] = [0, 20],
):

    # check whether exact intersecting cropped file already exists
    cmip6_dir_fp = config.cmip6_data_dir / source_id / member_id / "regridded"
    # TODO: include levs check. Much harder, so leaving for now
    correct_area_fps = list(
        find_files_for_area(cmip6_dir_fp.rglob("*.nc"), lat_range=lats, lon_range=lons),
    )
    # check that also spans full year range
    correct_fps = find_files_for_time(correct_area_fps, year_range=sorted(year_range))
    if len(correct_fps) > 0:
        # check that file includes all variables in variables list
        for fp in correct_fps:
            if all(variable in str(fp) for variable in variables):
                ds = cmipper_utils.process_xa_d(xa.open_dataset(fp))
                ds_years = ds["time.year"]
                return (
                    ds.sel(
                        time=(ds_years >= min(year_range))
                        & (ds_years <= max(year_range)),
                        latitude=slice(min(lats), max(lats)),
                        longitude=slice(min(lons), max(lons)),
                    )[variables],
                    fp,
                )
    return None, None

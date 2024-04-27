from pathlib import Path
import re
import xarray as xa

# from cmipper import utils, config
from coralshift.cmipper import utils, config


def find_files_for_time(filepaths, year_range):
    result = []
    for fp in filepaths:
        year = int(str(fp).split("_")[-1].split("-")[0][:4])
        if year_range[0] <= year <= year_range[1] - 1:
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

    # check whether intersecting cropped file already exists
    cmip6_dir_fp = (
        config.cmip6_data_dir / source_id / member_id / "testing" / "regridded"
    )
    # TODO: remove newtest
    # TODO: include levs check
    correct_area_fps = list(
        find_files_for_area(cmip6_dir_fp.rglob("*.nc"), lat_range=lats, lon_range=lons),
    )
    correct_fps = find_files_for_time(
        cmip6_dir_fp.rglob("*.nc"), year_range=sorted(year_range)
    )
    # TODO: check that also spans full year range
    if len(correct_fps) > 0:
        # check that file includes all variables in variables list
        for fp in correct_area_fps:
            if all(variable in str(fp) for variable in variables):
                return (
                    utils.process_xa_d(xa.open_dataset(fp)).sel(
                        latitude=slice(min(lats), max(lats)),
                        longitude=slice(min(lons), max(lons)),
                    ),
                    fp,
                )
    return None, None

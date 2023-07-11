from __future__ import annotations

import sys

# from pathlib import Path

# from . import dataloading


# # Get the parent directory of the current script
# current_dir = Path(__file__).resolve().parent
# print(current_dir)
# parent_dir = current_dir.parent

# # Add the parent directory to the Python path
# sys.path.insert(0, str(parent_dir))
from coralshift.dataloading import bathymetry

# import xarray
# import rasterio


# from coralshift.processing import spatial_data
# from coralshift.utils import file_ops, directories, utils


def display_area_options():
    print("Please choose a reef area from the list: ")
    print(
        "           A                        B                        C                        D"
    )
    print(
        " latitudes, longitudes |  latitudes, longitudes |  latitudes, longitudes |  latitudes, longitudes"
    )
    print(
        "(-10, -17), (142, 147) | (-16, -23), (144, 149) | (-18, -24), (148, 154) | (-23, -29), (150, 156)"
    )
    print("Your choice: ")


def display_resolution_options():
    print("Choose a resolution option.")
    print("'Native' resolutions are 1 or 1/12 degrees, or 4000m or 1000m.")


def main(
    area_name: str,
    target_resolution: float,
    target_resolution_unit: str,
    cmems_period: str = "monthly",
) -> None:
    # target_resolution_d = spatial_data.choose_resolution(
    #     target_resolution_tuple[0], target_resolution_tuple[1]
    # )
    # # Call the functions in sequence
    reef_areas = bathymetry.ReefAreas()
    area_name = reef_areas.get_name_from_names(area_name)
    print(area_name)
    print("target_Resolution:", target_resolution)
    print("target_Resolution unit:", target_resolution_unit)
    print("cmems: ", cmems_period)

    # # download bathymetry datarray via xarray
    # _, xa_bath = bathymetry.generate_bathymetry_xa_da(area_name)
    # # upsample to desired resolution
    # _, xa_bath_upsampled = spatial_data.upsample_and_save_xa_a(
    #     directories.get_bathymetry_datasets_dir(),
    #     xa_d=xa_bath,
    #     name=area_name,
    #     target_resolution_d=target_resolution_d,
    # )
    # # calculate sea floor gradients
    # grads, grads_path = bathymetry.generate_gradient_magnitude_nc(
    #     xa_bath_upsampled, sigma=1
    # )
    # # generate geojson file in reef_baseline directory for download from the Allen Coral Atlas
    # geojson_path = reef_extent.generate_area_geojson(
    #     area_class=reef_areas,
    #     area_name=area_name,
    #     save_dir=directories.get_reef_baseline_dir(),
    # )
    # # GEE PROCESSING

    # if cmems_period == "monthly":
    #     # Values generated here are those reported in the accompanying paper.
    #     xa_cmems_monthly, cmems_monthly_path = climate_data.download_reanalysis(
    #         download_dir=directories.get_monthly_cmems_dir(),
    #         final_filename="cmems_gopr_monthly",
    #         lat_lims=reef_areas.get_lat_lon_limits(area_name)[0],
    #         lon_lims=reef_areas.get_lat_lon_limits(area_name)[1],
    #         product_id="cmems_mod_glo_phy_my_0.083_P1M-m",
    #     )
    # elif cmems_period == "daily":
    #     # download daily data
    #     xa_cmems_daily, cmems_daily_path = climate_data.download_reanalysis(
    #         download_dir=directories.get_daily_cmems_dir(),
    #         final_filename="cmems_gopr_daily.nc",
    #         lat_lims=reef_areas.get_lat_lon_limits(area_name)[0],
    #         lon_lims=reef_areas.get_lat_lon_limits(area_name)[1],
    #         product_id="cmems_mod_glo_phy_my_0.083_P1D-m",
    #     )
    # else:
    #     print(f"CMEMS period option {cmems_period} not recognised.")
    # # download ERA5 data
    # climate_data.generate_era5_data(
    #     lat_lims=reef_areas.get_lat_lon_limits(area_name)[0],
    #     lon_lims=reef_areas.get_lat_lon_limits(area_name)[1],
    # )


if __name__ == "__main__":
    # only one argument supplied
    if len(sys.argv) < 2:
        print("Usage: python script_name area_name target_resolution [cmems_period]")
        sys.exit(1)

    area_name = sys.argv[1]
    target_resolution = sys.argv[2]
    target_resolution_unit = sys.argv[3]

    if len(sys.argv) >= 4:
        cmems_period = sys.argv[3]
    else:
        cmems_period = None

    main(area_name, target_resolution, target_resolution_unit, cmems_period)

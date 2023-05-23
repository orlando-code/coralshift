import numpy as np
import pandas as pd
import xarray as xa
import json
import os

# import getpass
import cdsapi

from pathlib import Path
from tqdm import tqdm
from pandas._libs.tslibs.timestamps import Timestamp

from coralshift.utils import utils, file_ops


def generate_spatiotemporal_var_filename_from_dict(
    info_dict: dict,
) -> str:
    """Generate a filename based on variable, date, and coordinate limits.

    Parameters
    ----------
    info_dict (dict): A dictionary containing information about the variable, date, and coordinate limits.

    Returns
    -------
    str: The generated filename.
    """
    filename_list = []
    for k, v in info_dict.items():
        # strings (variables)
        if utils.is_type_or_list_of_type(v, str):
            filename_list.extend([k.upper(), utils.underscore_str_of_strings(v)])
        # np.datetime64 (dates)
        elif utils.is_type_or_list_of_type(
            v, np.datetime64
        ) or utils.is_type_or_list_of_type(v, Timestamp):
            filename_list.extend([k.upper(), utils.underscore_str_of_dates(v)])
        # tuples (coordinates limits)
        elif utils.is_type_or_list_of_type(v, tuple):
            filename_list.extend(
                [k.upper(), utils.underscore_list_of_tuples(utils.round_list_tuples(v))]
            )
    return "_".join(filename_list)


def generate_metadata(
    download_dir: str,
    filename: str,
    variable: str,
    date_lims: tuple[str, str],
    lon_lims: list[float],
    lat_lims: list[float],
    depth_lims: list[float],
    query: str,
) -> None:
    """Generate metadata for the downloaded file and save it as a JSON file.

    Parameters
    ----------
    download_dir (str): The directory where the file is downloaded.
    filename (str): The name of the file.
    variable (str): The variable acronym.
    date_lims (tuple[str, str]): A tuple containing the start and end dates.
    lon_lims (list[float]): A list containing the longitude limits.
    lat_lims (list[float]): A list containing the latitude limits.
    depth_lims (list[float]): A list containing the depth limits.
    query (str): The MOTU query used for downloading the file.
    """
    filepath = (Path(download_dir) / filename).with_suffix(".json")

    var_dict = {
        "mlotst": "ocean mixed layer thickness (sigma theta)",
        "siconc": "sea ice area fraction",
        "thetao": "sea water potential temperature",
        "usi": "eastward sea ice velocity",
        "sithick": "sea ice thickness",
        "bottomT": "sea water potential at sea floor",
        "vsi": "northward sea ice velocity",
        "usi": "eastward sea ice velocity",
        "vo": "northward sea water velocity",
        "uo": "eastward sea water velocity",
        "so": "sea water salinity",
        "zos": "sea surface height above geoid",
    }

    metadata = {
        "filename": filename,
        "download directory": str(download_dir),
        "variable acronym": variable,
        "variable name": var_dict[variable],
        "longitude-min": lon_lims[0],
        "longitude-max": lon_lims[1],
        "latitude-min": lat_lims[0],
        "latitude-max": lat_lims[1],
        "date-min": str(date_lims[0]),
        "date-max": str(date_lims[1]),
        "depth-min": depth_lims[0],
        "depth-max": depth_lims[1],
        "motu_query": query,
    }

    # Serializing json
    json_object = json.dumps(metadata, indent=4)

    with open(str(filepath), "w") as outfile:
        outfile.write(json_object)


def generate_name_dict(
    variables: list[str],
    date_lims: tuple[str, str],
    lon_lims: tuple[str, str],
    lat_lims: tuple[str, str],
    depth_lims: tuple[str, str],
) -> dict:
    return {
        "vars": variables,
        "dates": date_lims,
        "lons": lon_lims,
        "lats": lat_lims,
        "depths": depth_lims,
    }


def download_reanalysis(
    download_dir: str | Path,
    variables: list[str],
    date_lims: tuple[str, str],
    lon_lims: tuple[str, str],
    lat_lims: tuple[str, str],
    depth_lims: tuple[str, str],
    product_type: str = "my",
    product_id: str = "GLOBAL_MULTIYEAR_PHY_001_030",
    dataset_id: str = "cmems_mod_glo_phy_my_0.083_P1D-m",
) -> xa.Dataset:
    """
    Download reanalysis data for multiple variables and save them to the specified directory.

    Parameters
    ----------
    download_dir (str | Path): Directory to save the downloaded files.
    variables (list[str]): List of variables to download.
    date_lims (tuple[str, str]): Date limits as a tuple of strings in the format "YYYY-MM-DD".
    lon_lims (tuple[str, str]): Longitude limits as a tuple of strings in the format "lon_min, lon_max".
    lat_lims (tuple[str, str]): Latitude limits as a tuple of strings in the format "lat_min, lat_max".
    depth_lims (tuple[str, str]): Depth limits as a tuple of strings in the format "depth_min, depth_max".
    product_type (str, optional): Product type. Defaults to "my".
    product_id (str, optional): Product ID. Defaults to "GLOBAL_MULTIYEAR_PHY_001_030".
    dataset_id (str, optional): Dataset ID. Defaults to "cmems_mod_glo_phy_my_0.083_P1D-m".

    Returns
    -------
    xa.Dataset: dataset merged from individual files
    """
    download_dir = file_ops.guarantee_existence(download_dir)

    # User credentials
    # username = input("Enter your username: ")
    # password = getpass.getpass("Enter your password: ")
    username = "otimmerman"
    password = "Fgg0N$tUUuL3"

    # split request by variable
    for var in tqdm(variables, desc=" variable loop", position=0):
        print(f"Downloading {var}...")
        # split request by time
        date_pairs = utils.generate_date_pairs(date_lims)
        # create download folder for each variable (if not already existing)
        save_dir = Path(download_dir) / var
        file_ops.guarantee_existence(save_dir)
        for sub_date_lims in tqdm(date_pairs):
            # generate name info dictionary
            name_dict = generate_name_dict(
                var, sub_date_lims, lon_lims, lat_lims, depth_lims
            )
            filename = generate_spatiotemporal_var_filename_from_dict(name_dict)
            # if file doesn't already exist, generate and execute API query
            # print((Path(save_dir) / filename).with_suffix(".nc"))
            if not (Path(save_dir) / filename).with_suffix(".nc").is_file():
                query = generate_motu_query(
                    save_dir,
                    filename,
                    var,
                    sub_date_lims,
                    lon_lims,
                    lat_lims,
                    depth_lims,
                    product_type,
                    product_id,
                    dataset_id,
                    username,
                    password,
                )
                execute_motu_query(
                    save_dir,
                    filename,
                    var,
                    sub_date_lims,
                    lon_lims,
                    lat_lims,
                    depth_lims,
                    query,
                )
            else:
                print(f"{filename} already exists in {save_dir}.")

    # generate name of combined file
    name_dict = generate_name_dict(variables, date_lims, lon_lims, lat_lims, depth_lims)
    main_filename = generate_spatiotemporal_var_filename_from_dict(name_dict)
    save_path = (Path(download_dir) / main_filename).with_suffix(".nc")
    merged_nc = file_ops.load_merge_nc_files(download_dir)

    merged_nc.to_netcdf(save_path)
    print(f"Combined nc file written to {save_path}.")

    return merged_nc


def execute_motu_query(
    download_dir: str | Path,
    filename: str,
    var: list[str] | str,
    sub_date_lims: tuple[np.datetime64, np.datetime64],
    lon_lims: tuple[float, float],
    lat_lims: tuple[float, float],
    depth_lims: tuple[float, float],
    query: str,
) -> None:
    """Execute the MOTU query to download the data slice and generate metadata.

    Parameters
    ----------
    download_dir (str | Path): The directory where the file will be downloaded.
    filename (str): The name of the file.
    var (list[str] | str): The variable(s) to be downloaded.
    sub_date_lims (tuple[np.datetime64, np.datetime64]): A tuple containing the start and end dates.
    lon_lims (tuple[float, float]): A tuple containing the longitude limits.
    lat_lims (tuple[float, float]): A tuple containing the latitude limits.
    depth_lims (tuple[float, float]): A tuple containing the depth limits.
    query (str): The MOTU query used for downloading the file.

    Returns
    -------
    None
    """
    outcome = download_data_slice(query)
    # if download successful
    if outcome:
        generate_metadata(
            download_dir,
            filename,
            var,
            sub_date_lims,
            lon_lims,
            lat_lims,
            depth_lims,
            query,
        )
        print(f"{filename} written to {download_dir} and metadata generated.")


def generate_motu_query(
    download_dir: str | Path,
    filename: str,
    variable: list[str] | str,
    date_lims: tuple[np.datetime64, np.datetime64],
    lon_lims: tuple[float, float],
    lat_lims: tuple[float, float],
    depth_lims: tuple[float, float],
    product_type: str,
    product_id: str,
    dataset_id: str,
    username: str,
    password: str,
) -> str:
    """Generate the MOTU query for downloading climate data.

    Parameters
    ----------
    download_dir (str | Path): The directory where the file will be downloaded.
    filename (str): The name of the file.
    variable (list[str] | str): The variable(s) to be downloaded.
    date_lims (tuple[np.datetime64, np.datetime64]): A tuple containing the start and end dates.
    lon_lims (tuple[float, float]): A tuple containing the longitude limits.
    lat_lims (tuple[float, float]): A tuple containing the latitude limits.
    depth_lims (tuple[float, float]): A tuple containing the depth limits.
    product_type (str): The type of product.
    product_id (str): The product ID.
    dataset_id (str): The dataset ID.
    username (str): The username for authentication.
    password (str): The password for authentication.

    Returns
    -------
    query (str): The MOTU query.

    """

    lon_min, lon_max = min(lon_lims), max(lon_lims)
    lat_min, lat_max = min(lat_lims), max(lat_lims)
    date_min, date_max = min(date_lims), max(date_lims)
    depth_min, depth_max = min(depth_lims), max(depth_lims)

    # Generate motuclient command line
    query = f"python -m motuclient --motu https://{product_type}.cmems-du.eu/motu-web/Motu \
    --service-id {product_id}-TDS --product-id {dataset_id} \
    --longitude-min {lon_min} --longitude-max {lon_max} --latitude-min {lat_min} --latitude-max {lat_max} \
    --date-min '{date_min}' --date-max '{date_max}' --depth-min {depth_min} --depth-max {depth_max} \
    --variable {variable} --out-dir '{download_dir}' --out-name '{filename}.nc' --user '{username}' --pwd '{password}'"

    return query


def download_data_slice(query):
    """Download the data slice using the MOTU query.

    Parameters
    ----------
    query (str): The MOTU query.

    Returns
    -------
    bool: True if the download is successful, False otherwise.
    """
    try:
        os.system(query)
        return True
    except ConnectionAbortedError():
        print("Data download failed.")
        return False


def ecmwf_api_call(
    c,
    download_dest_dir: Path | str,
    parameter: str,
    time_info_dict: dict,
    area: list[tuple[float]],
    format: str,
):
    api_call_dict = generate_ecmwf_api_dict(parameter, time_info_dict, area, format)
    # make api call
    try:
        # TODO: update this
        c.retrieve("reanalysis-era5-land", api_call_dict, download_dest_dir)
    # if error in fetching, limit the parameter
    except ConnectionAbortedError():
        print(f"API call failed for {parameter}.")


def generate_ecmwf_api_dict(
    weather_params: list[str], time_info_dict: dict, area: list[float], format: str
) -> dict:
    """Generate api dictionary format for single month of event"""

    api_call_dict = {
        "variable": weather_params,
        "area": area,
        "format": format,
    } | time_info_dict

    return api_call_dict


def return_full_ecmwf_weather_param_strings(dict_keys: list[str]):
    """Look up weather parameters in a dictionary so they can be entered as short strings rather than typed out in full.
    Key:value pairs ordered in expected importance

    Parameters
    ----------
    dict_keys : list[str]
        list of shorthand keys for longhand weather parameters. See accompanying documentation on GitHub
    """

    weather_dict = {
        "d2m": "2m_dewpoint_temperature",
        "t2m": "2m_temperature",
        "skt": "skin_temperature",
        "tp": "total_precipitation",
        "sp": "surface_pressure",
        "src": "skin_reservoir_content",
        "swvl1": "volumetric_soil_water_layer_1",
        "swvl2": "volumetric_soil_water_layer_2",
        "swvl3": "volumetric_soil_water_layer_3",
        "swvl4": "volumetric_soil_water_layer_4",
        "slhf": "surface_latent_heat_flux",
        "sshf": "surface_sensible_heat_flux",
        "ssr": "surface_net_solar_radiation",
        "str": "surface_net_thermal_radiation",
        "ssrd": "surface_solar_radiation_downwards",
        "strd": "surface_thermal_radiation_downwards",
        "e": "total_evaporation",
        "pev": "potential_evaporation",
        "ro": "runoff",
        "ssro": "sub-surface_runoff",
        "sro": "surface_runoff",
        "u10": "10m_u_component_of_wind",
        "v10": "10m_v_component_of_wind",
    }

    weather_params = []
    for key in dict_keys:
        weather_params.append(weather_dict.get(key))

    return weather_params


def generate_times_from_start_end(start_end_dates: list[tuple[pd.Timestamp]]) -> dict:
    """Generate dictionary containing ecmwf time values from list of start and end dates.

    TODO: update so can span multiple months accurately (will involve several api calls)
    """

    # padding dates of interest + 1 day on either side to deal with later nans
    dates = pd.date_range(
        start_end_dates[0] - pd.Timedelta(1, "d"),
        start_end_dates[1] + pd.Timedelta(1, "d"),
    )
    years, months, days, hours = set(), set(), set(), []
    # extract years from time
    for date in dates:
        years.add(str(date.year))
        months.add(utils.pad_number_with_zeros(date.month))
        days.add(utils.pad_number_with_zeros(date.day))

    for i in range(24):
        hours.append(f"{i:02d}:00")

    years, months, days = list(years), list(months), list(days)

    time_info = {"year": years, "month": months[0], "day": days, "time": hours}

    return time_info


def fetch_era5_data(
    weather_params: list[str],
    date_lims: tuple[np.datetime64, np.datetime64],
    # areas: list[tuple[float]],
    lon_lims: tuple[float, float],
    lat_lims: tuple[float, float],
    download_dest_dir: str | Path,
    format: str = "grib",
) -> None:
    """Generate API call, download files, merge xarrays, save as new pkl file.

    Parameters
    ----------
    weather_keys : list[str]
        list of weather parameter short names to be included in the call
    start_end_dates : list[tuple[pd.Timestamp]]
        list of start and end date/times for each event
    area : list[tuple[float]]
        list of max/min lat/lon values in format [north, west, south, east]
    download_dest_dir : str | Path
        path to download destination
    format : str = 'grib'
        format of data file to be downloaded

    Returns
    -------
    None
    """
    # initialise client
    c = cdsapi.Client()
    # for parameter in weather_params
    for param in weather_params:
        download_dir_name = file_ops.guarantee_existence(
            Path(download_dest_dir) / param
        )
        # for month in range of time start and finish
        for month in pd.date_range(date_lims[0], date_lims[1], freq="MS"):
            # generate time range TODO
            time_info_dict = generate_times_from_start_end(month)
            # N.B. this will result in unexpected behaviour for negative values
            area = [max(lat_lims), min(lon_lims), min(lat_lims), max(lon_lims)]

            filename = f"{param}_{str(month)}.{format}"
            filepath = file_ops.generate_filepath(download_dir_name, filename, format)
            ecmwf_api_call(c, download_dir_name, param, time_info_dict, area)

    # for i, dates in enumerate(start_end_dates):
    #     # create new folder for downloads - TODO: FUNCTION
    #     dir_name = "_".join(utils.dates_from_dt(dates))
    #     dir_path = file_ops.guarantee_existence(Path(download_dest_dir) / dir_name)
    #     # dir_name = '_'.join((
    #     #     dates[0].strftime("%d-%m-%Y"), dates[1].strftime("%d-%m-%Y")
    #     #     ))
    #     # dir_path = guarantee_existence(os.path.join(download_dest_dir, dir_name))

    #     time_info_dict = generate_times_from_start_end(dates)

    #     for param in weather_params:
    #         # generate api call info TODO: FUNCTION
    #         filename = f"{param}.{format}"
    #         filepath = file_ops.generate_filepath(download_dest_dir, filename, format)
    #         ecmwf_api_call(c, download_dest_dir, param, time_info_dict, areas[i])

    # api_call_dict = generate_api_dict(param, time_info_dict, areas[i], format)
    # file_name = f'{param}.{format}'
    # dest = '/'.join((dir_path, file_name))
    # # make api call
    # try:
    #     c.retrieve(
    #         'reanalysis-era5-land',
    #         api_call_dict,
    #         dest
    #     )
    # # if error in fetching, limit the parameter
    # except TypeError():
    #     print(f'{param} not found in {dates}. Skipping fetching, moving on.')

    # TODO: FUNCTION

    # load in all files in folder

    # filepath = climate_data.generate_spatiotemporal_var_filename(weather_params, dates, area[0], area[1],)
    # save as new file

    # file_paths = file_ops.return_list_filepaths(download_dest_dir, suffix)

    # xa_dict = {}
    # for file_path in tqdm(glob.glob(file_paths)):
    #     # get name of file
    #     file_name = file_path.split('/')[-1]
    #     # read into xarray
    #     xa_dict[file_name] = xr.load_dataset(file_path, engine="cfgrib")

    # # merge TODO: apparently conflicting values of 'step'. Unsure why.
    # out = xr.merge([array for array in xa_dict.values()], compat='override')
    # # save as new file
    # nc_file_name = '.'.join((dir_name, 'nc'))
    # save_file_path = '/'.join((download_dest_dir, nc_file_name))
    # out.to_netcdf(path=save_file_path)
    # print(f'{nc_file_name} saved successfully')

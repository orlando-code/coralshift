from __future__ import annotations

import numpy as np
import xarray as xa
import os
import cdsapi

# import getpass

# import cdsapi

from pathlib import Path
from tqdm import tqdm
from pandas._libs.tslibs.timestamps import Timestamp

from coralshift.utils import utils, file_ops, directories
from coralshift.processing import spatial_data
from coralshift.dataloading import bathymetry


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
            filename_list.extend(
                [
                    k.upper(),
                    utils.replace_dot_with_dash(utils.underscore_str_of_strings(v)),
                ]
            )
        # np.datetime64 (dates)
        elif utils.is_type_or_list_of_type(
            v, np.datetime64
        ) or utils.is_type_or_list_of_type(v, Timestamp):
            filename_list.extend(
                [
                    k.upper(),
                    utils.replace_dot_with_dash(utils.underscore_str_of_dates(v)),
                ]
            )
        # tuples (coordinates limits)
        elif utils.is_type_or_list_of_type(v, tuple):
            filename_list.extend(
                [
                    k.upper(),
                    utils.replace_dot_with_dash(
                        utils.underscore_list_of_tuples(utils.round_list_tuples(v))
                    ),
                ]
            )
    return "_".join(filename_list)


def generate_metadata(
    download_dir: str,
    filename: str,
    variables: list[str],
    date_lims: tuple[str, str],
    lon_lims: list[float],
    lat_lims: list[float],
    depth_lims: list[float],
    query: str = "n/a",
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
        "bottomT": "sea water potential temperature at sea floor",
        "vsi": "northward sea ice velocity",
        "usi": "eastward sea ice velocity",
        "vo": "northward sea water velocity",
        "uo": "eastward sea water velocity",
        "so": "sea water salinity",
        "zos": "sea surface height above geoid",
    }

    # if list of variables, iterate through dict to get long names
    if len(variables) > 1:
        variable_names = str([var_dict[var] for var in variables])
    else:
        variable_names = var_dict[variables[0]]
    # send list to a string for json
    variables = str(variables)

    metadata = {
        "filename": filename,
        "download directory": str(download_dir),
        "variable acronym(s)": variables,
        "variable name(s)": variable_names,
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
    # save metadata as json file at filepath
    file_ops.save_json(metadata, filepath)


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
    region: str,
    final_filename: str = None,
    variables: list[str] = ["mlotst", "bottomT", "uo", "so", "zos", "thetao", "vo"],
    date_lims: tuple[str, str] = ("1992-12-31", "2020-12-16"),
    depth_lims: tuple[str, str] = (0.3, 1),
    lon_lims: tuple[str, str] = (142, 147),
    lat_lims: tuple[str, str] = (-17, -10),
    product_type: str = "my",
    service_id: str = "GLOBAL_MULTIYEAR_PHY_001_030",
    product_id: str = "cmems_mod_glo_phy_my_0.083_P1D-m",
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
    service_id (str, optional): Product ID. Defaults to "GLOBAL_MULTIYEAR_PHY_001_030".
    product_id (str, optional): Dataset ID. Defaults to "cmems_mod_glo_phy_my_0.083_P1D-m".

    Returns
    -------
    xa.Dataset: dataset merged from individual files

    Notes
    -----
    Currently taking only topmost depth (TODO: make full use of profile)

    """
    download_dir = file_ops.guarantee_existence(Path(download_dir) / region)
    merged_download_dir = file_ops.guarantee_existence(download_dir / "merged_vars")

    # User credentials
    # username = input("Enter your username: ")
    # password = getpass.getpass("Enter your password: ")
    username = "otimmerman"
    password = "Fgg0N$tUUuL3"

    # generate name of combined file
    name_dict = generate_name_dict(variables, date_lims, lon_lims, lat_lims, depth_lims)
    main_filename = generate_spatiotemporal_var_filename_from_dict(name_dict)
    save_path = (Path(download_dir) / main_filename).with_suffix(".nc")

    # if particular filename specified
    if final_filename:
        save_path = (
            Path(download_dir) / file_ops.remove_suffix(final_filename)
        ).with_suffix(".nc")

    if save_path.is_file():
        print(f"Merged file already exists at {save_path}")
        return xa.open_dataset(save_path), save_path

    date_merged_xas = []
    # split request by variable
    for var in tqdm(variables, desc=" variable loop", position=0, leave=True):
        print(f"Downloading {var} data...")
        # split request by time
        date_pairs = utils.generate_date_pairs(date_lims)
        # create download folder for each variable (if not already existing)
        save_dir = Path(download_dir) / var
        file_ops.guarantee_existence(save_dir)
        for sub_date_lims in tqdm(date_pairs, leave=False):
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
                    service_id,
                    product_id,
                    username,
                    password,
                )
                execute_motu_query(
                    save_dir,
                    filename,
                    [var],
                    sub_date_lims,
                    lon_lims,
                    lat_lims,
                    depth_lims,
                    query,
                )
            else:
                print(f"{filename} already exists in {save_dir}.")

        var_name_dict = generate_name_dict(
            var, date_lims, lon_lims, lat_lims, depth_lims
        )
        date_merged_name = generate_spatiotemporal_var_filename_from_dict(var_name_dict)
        # merge files by time
        merged_path = file_ops.merge_nc_files_in_dir(
            save_dir,
            date_merged_name,
            merged_save_path=(merged_download_dir / date_merged_name).with_suffix(
                ".nc"
            ),
        )

        # generate metadata if necessary
        if not (merged_download_dir / date_merged_name).with_suffix(".json").is_file():
            generate_metadata(
                merged_download_dir,
                date_merged_name,
                [var],
                date_lims,
                lon_lims,
                lat_lims,
                depth_lims,
            )

        date_merged_xas.append(merged_path)

    # concatenate variables
    arrays = [
        (
            spatial_data.process_xa_d(
                xa.open_dataset(date_merged_xa, decode_coords="all")
            )
        )
        for date_merged_xa in date_merged_xas
    ]
    all_merged = xa.merge(arrays)

    save_nc(download_dir, final_filename, all_merged)
    # all_merged.to_netcdf(save_path)
    # generate accompanying metadata
    generate_metadata(
        download_dir,
        final_filename,
        variables,
        date_lims,
        lon_lims,
        lat_lims,
        depth_lims,
    )
    return all_merged, save_path


def save_nc(
    save_dir: Path | str,
    filename: str,
    xa_d: xa.DataArray | xa.Dataset,
    return_array: bool = False,
) -> xa.DataArray | xa.Dataset:
    """
    Save the given xarray DataArray or Dataset to a NetCDF file iff no file with the same
    name already exists in the directory.
    # TODO: issues when suffix provided
    Parameters
    ----------
        save_dir (Path or str): The directory path to save the NetCDF file.
        filename (str): The name of the NetCDF file.
        xa_d (xarray.DataArray or xarray.Dataset): The xarray DataArray or Dataset to be saved.

    Returns
    -------
        xarray.DataArray or xarray.Dataset: The input xarray object.
    """
    filename = file_ops.remove_suffix(utils.replace_dot_with_dash(filename))
    save_path = (Path(save_dir) / filename).with_suffix(".nc")
    if not save_path.is_file():
        if "grid_mapping" in xa_d.attrs:
            del xa_d.attrs["grid_mapping"]
        print(f"Writing {filename} to file at {save_path}")
        spatial_data.process_xa_d(xa_d).to_netcdf(save_path)
        print("Writing complete.")
    else:
        print(f"{filename} already exists in {save_dir}")

    if return_array:
        return save_path, xa.open_dataset(save_path)
    else:
        return save_path


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
    service_id: str,
    product_id: str,
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
    service_id (str): The product ID.
    product_id (str): The dataset ID.
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

    # generate motuclient command line
    # specifying environment to conda environment makes sure that `motuclient` module is found
    query = f"conda run -n coralshift python -m motuclient \
    --motu https://{product_type}.cmems-du.eu/motu-web/Motu --service-id {service_id}-TDS --product-id {product_id} \
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
    filepath: str,
    parameter: str,
    time_info_dict: dict,
    area: list[tuple[float]],
    dataset_tag: str = "reanalysis-era5-single-levels",
    format: str = "nc",
):
    api_call_dict = generate_ecmwf_api_dict(parameter, time_info_dict, area, format)
    # make api call
    try:
        c.retrieve(dataset_tag, api_call_dict, filepath)
    # if error in fetching, limit the parameter
    except ConnectionAbortedError():
        print(f"API call failed for {parameter}.")


def generate_ecmwf_api_dict(
    weather_params: list[str], time_info_dict: dict, area: list[float], format: str
) -> dict:
    """Generate api dictionary format for single month of event"""

    # if weather_params

    api_call_dict = {
        "product_type": "reanalysis",
        "variable": [weather_params],
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


def hourly_means_to_daily(hourly_dir: Path | str, suffix: str = "netcdf"):
    filepaths = file_ops.return_list_filepaths(hourly_dir, suffix, incl_subdirs=True)
    # create subdirectory to store averaged files
    daily_means_dir = file_ops.guarantee_existence(Path(hourly_dir) / "daily_means")
    for filepath in tqdm(filepaths, desc="Converting hourly means to daily means"):
        filename = "_".join((str(filepath.stem), "daily"))
        save_path = (daily_means_dir / filename).with_suffix(
            file_ops.pad_suffix(suffix)
        )
        # open dataset
        hourly = xa.open_dataset(filepath, chunks={"time": 100})
        daily = hourly.resample(time="1D").mean()
        # take average means
        daily.to_netcdf(save_path)


def generate_month_day_hour_list(items_range):
    items = []

    if isinstance(items_range, (int, np.integer)):
        items_range = [items_range]
    elif isinstance(items_range, np.ndarray):
        items_range = items_range.tolist()
    elif not isinstance(items_range, list):
        raise ValueError(
            "Invalid input format. Please provide an integer, a list, or a NumPy array."
        )

    for item in items_range:
        if isinstance(item, (int, np.integer)):
            if item < 0 or item > 31:
                raise ValueError("Invalid items value: {}.".format(item))
            items.append(item)
        else:
            raise ValueError(
                "Invalid input format. Please provide an integer, a list, or a NumPy array."
            )

    return items


def return_times_info(
    year: int,
    months: list[int] | int = np.arange(1, 13),
    days: list[int] | int = np.arange(1, 32),
    hours: list[int] | int = np.arange(0, 24),
):
    year = str(year)
    months = [
        utils.pad_number_with_zeros(month)
        for month in generate_month_day_hour_list(months)
    ]
    days = [
        utils.pad_number_with_zeros(day) for day in generate_month_day_hour_list(days)
    ]

    hours = [
        utils.pad_number_with_zeros(hour)
        for hour in generate_month_day_hour_list(hours)
    ]
    for h, hour in enumerate(hours):
        hours[h] = f"{hour}:00"

    return {"year": year, "month": months, "day": days, "time": hours}


def fetch_weather_data(
    download_dest_dir,
    weather_params,
    years,
    months: list[int] | int = np.arange(1, 13),
    days: list[int] | int = np.arange(1, 32),
    hours: list[int] | int = np.arange(0, 24),
    lat_lims=(-10, -17),
    lon_lims=(142, 147),
    dataset_tag: str = "reanalysis-era5-single-levels",
    format: str = "netcdf",
):
    c = cdsapi.Client()

    area = [max(lat_lims), min(lon_lims), min(lat_lims), max(lon_lims)]

    for param in weather_params:
        param_download_dest = file_ops.guarantee_existence(
            Path(download_dest_dir) / param
        )
        for year in years:
            filename = generate_spatiotemporal_var_filename_from_dict(
                {"var": param, "lats": lat_lims, "lons": lon_lims, "year": str(year)}
            )
            # filename = str(file_ops.generate_filepath(param_download_dest, filename, format))
            filepath = str(
                file_ops.generate_filepath(param_download_dest, filename, format)
            )

            if not Path(filepath).is_file():
                time_info_dict = return_times_info(year, months, days)
                # filename = str(file_ops.generate_filepath(param_download_dest, f"{param}_{year}", format))
                # filename = str((param_download_dest / param / str(year)).with_suffix(format))
                ecmwf_api_call(
                    c, filepath, param, time_info_dict, area, dataset_tag, format
                )
            else:
                print(f"Filepath already exists: {filepath}")
        # TODO: more descriptive filename


def generate_era5_data(
    weather_params: list[float] = [
        "evaporation",
        "significant_height_of_combined_wind_waves_and_swell",
        "surface_net_solar_radiation",
        "surface_pressure",
    ],
    region: str = None,
    years: list[int] = np.arange(1993, 2021),
    lat_lims: tuple[float] = (-10, -17),
    lon_lims: tuple = (142, 147),
) -> None:
    """
    Generates and merges ERA5 weather data files.

    Parameters
    ----------
        weather_params (list[str]): A list of weather parameters to download and merge.
        years (list[int]): A list of years for which to download and merge the data.
        lat_lims (tuple[float, float]): A tuple specifying the latitude limits.
        lon_lims (tuple[float, float]): A tuple specifying the longitude limits.

    Returns
    -------
    None
    """
    save_dir = directories.get_era5_data_dir()
    if region:
        save_dir = save_dir / bathymetry.ReefAreas().get_short_filename(region)
        (lat_lims, lon_lims) = bathymetry.ReefAreas().get_lat_lon_limits(region)

    # download data to appropriate folder(s)
    fetch_weather_data(
        download_dest_dir=save_dir,
        weather_params=weather_params,
        years=years,
        format="netcdf",
    )

    # combine files in folder to single folder
    combined_save_dir = file_ops.guarantee_existence(save_dir / "weather_parameters")
    print("\n")
    for param in weather_params:
        # get path to unmerged files
        param_dir = save_dir / param
        merged_name = generate_spatiotemporal_var_filename_from_dict(
            {
                "var": param,
                "lats": lat_lims,
                "lons": lon_lims,
                "year": f"{str(years[0])}-{str(years[-1])}",
            }
        )
        # generate combined save path
        combined_save_path = (combined_save_dir / merged_name).with_suffix(".nc")

        file_ops.merge_nc_files_in_dir(
            nc_dir=param_dir, merged_save_path=combined_save_path, format=".netcdf"
        )

    print(
        f"\nAll ERA5 weather files downloaded by year and merged into {combined_save_dir}"
    )

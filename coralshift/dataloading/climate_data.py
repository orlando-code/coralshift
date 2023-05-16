import numpy as np
import xarray as xa
import json
import os

# import getpass

from pathlib import Path
from tqdm import tqdm

from coralshift.utils import utils, file_ops


def generate_spatiotemporal_var_filename(
    variable: str | list[str],
    date_lims: tuple[np.datetime64, np.datetime64],
    lon_lims: list[tuple[float]],
    lat_lims: list[tuple[float]],
    depth_lims: list[tuple[float]],
) -> str:
    """Generate a filename based on variable, date, and coordinate limits.

    Parameters
    ----------
    variable (str | list[str]): The variable(s) as a string or a list of variables.
    date_lims (tuple[datetime, datetime]): A tuple of start and end datetime objects.
    lon_lims (list[tuple[float]]): A list of tuples representing longitude limits.
    lat_lims (list[tuple[float]]): A list of tuples representing latitude limits.
    depth_lims (list[tuple[float]]): A list of tuples representing depth limits.

    Returns:
    --------
    str: The generated filename.
    """
    vars = utils.vars_to_strs(variable)
    date_lims = utils.dates_from_dt(date_lims)
    lon_lims, lat_lims, depth_lims = utils.round_list_tuples(
        [lon_lims, lat_lims, depth_lims]
    )
    date_str = f"dts_{date_lims[0]}_{date_lims[1]}"
    coord_str = f"lon_{lon_lims[0]}_{lon_lims[1]}_lat_{lat_lims[0]}_{lat_lims[1]}_dep_{depth_lims[0]}_{depth_lims[1]}"
    return "_".join((f"{vars}", date_str, coord_str)).replace(".", "-")


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
        "download directory": download_dir,
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
    print(filename)

    with open(filepath, "w") as outfile:
        outfile.write(json_object)


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
        # split request by time
        date_pairs = utils.generate_date_pairs(date_lims)
        # print(date_pairs)
        for sub_date_lims in tqdm(date_pairs):
            filename = generate_spatiotemporal_var_filename(
                var, sub_date_lims, lon_lims, lat_lims, depth_lims
            )
            if not (Path(download_dir) / filename).with_suffix(".nc").is_file():
                query = generate_motu_query(
                    download_dir,
                    filename,
                    var,
                    date_lims,
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
                    download_dir,
                    filename,
                    var,
                    sub_date_lims,
                    lon_lims,
                    lat_lims,
                    depth_lims,
                    query,
                )
            else:
                print(f"{filename} already exists in {download_dir}.")
        print("Moving on to next variable...")

    main_filename = generate_spatiotemporal_var_filename(
        variables, date_lims, lon_lims, lat_lims, depth_lims
    )

    return file_ops.merge_save_nc_files(download_dir, main_filename)


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

# file handling
from pathlib import Path
import requests
import xarray as xa
import re
import numpy as np


# download
from concurrent.futures import ThreadPoolExecutor
import subprocess

# from cdo import Cdo

# custom
from cmipper import config, utils


# TODO: get this checking as a gateway to the main downloading script. But not urgent
def ensure_cmip6_downloaded(
    variables: list[str],
    source_id: str = "EC-Earth3P-HR",
    member_id: str = "r1i1p2f1",
    logging_dir: str = config.logging_dir,
):
    if not config.cmip6_data_dir.exists():
        config.cmip6_data_dir.mkdir(parents=True, exist_ok=True)

    potential_ds, potential_ds_fp = find_intersecting_cmip(
        variables=variables,
        source_id=source_id,
        member_id=member_id,
        lats=lats,
        lons=lons,
        year_range=year_range,
        levs=levs,
    )
    # if some files are missing, initialise download    TODO: be more circumspect about which downloads
    if potential_ds is None:
        if not Path(logging_dir).exists():
            Path(logging_dir).mkdir(parents=True, exist_ok=True)

        print("Creating/downloading necessary file(s)...")
        # TODO: add in other sources/members

    else:
        print(
            f"CMIP6 file with necessary variables spanning latitudes {lats} and longitudes {lons} already exists at: \n{potential_ds_fp}"  # noqa
        )


# DEPRECATED
################################################################################


# def execute_subprocess_command(command, output_log_path):
#     try:
#         result = subprocess.run(
#             command,
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE,
#             check=True,
#         )
#         print(f"{output_log_path}")
#         with open(output_log_path, "w") as output_log_file:
#             output_log_file.write(result.stdout.decode())
#             output_log_file.write(result.stderr.decode())

#         # with open(str(output_log_path), "w") as error_log_file:
#         #     error_log_file.write(result.stderr.decode())

#     except subprocess.CalledProcessError as e:
#         print(f"Error{e}")
#         # # Handle the exception if needed
#         with open(output_log_path, "w") as error_log_file:
#             error_log_file.write(f"Error: {e}")
#             error_log_file.write(result.stderr.decode())


# def run_commands_with_thread_pool(cmip_commands, output_log_paths, error_log_paths):
#     with ThreadPoolExecutor(max_workers=len(cmip_commands)) as executor:
#         # Submit each script command to the executor
#         executor.map(execute_subprocess_command, cmip_commands, output_log_paths)


# def construct_cmip_commands(
#     variables,
#     source_id: str = "EC-Earth3P-HR",
#     member_id: str = "r1i1p2f1",
#     lats: list[float, float] = [-40, 0],
#     lons: list[float, float] = [130, 170],
#     levs: list[int, int] = [0, 20],
#     year_range: list[int, int] = [1950, 2014],
#     logging_dir: str = config.logging_dir,
# ) -> (list, list):
#     cmip_commands = []
#     output_log_paths = []

#     for variable in variables:
#         # if not, run necessary downloading
#         cmip_command = construct_cmip_command(
#             variable=variable,
#             source_id=source_id,
#             member_id=member_id,
#             lats=lats,
#             lons=lons,
#             levs=levs,
#             year_range=year_range,
#         )
#         cmip_commands.append(cmip_command)

#         output_log_paths.append(
#             Path(logging_dir) / f"{source_id}_{member_id}_{variable}.txt"
#         )
#     return cmip_commands, output_log_paths


# def construct_cmip_command(
#     variable: str,
#     source_id: str = "EC-Earth3P-HR",
#     member_id: str = "r1i1p2f1",
#     lats: list[float, float] = [-40, 0],
#     lons: list[float, float] = [130, 170],
#     levs: list[int, int] = [0, 20],
#     year_range: list[int, int] = [1950, 2014],
#     script_fp: str = config.get_cmipper_module_dir()
#     / "download_cmip6_data_parallel.py",
# ):
#     arguments = [
#         "--source_id",
#         source_id,
#         "--member_id",
#         member_id,
#         "--variable_id",
#         variable,
#     ]

#     # Define the list arguments with their corresponding values
#     list_arguments = [
#         ("--lats", lats),
#         ("--lons", lons),
#         ("--levs", levs),
#         ("--year_range", year_range),
#     ]
#     list_str = [
#         item
#         for sublist in [
#             [flag, str(val)] for flag, vals in list_arguments for val in vals
#         ]
#         for item in sublist
#     ]

#     arguments += list_str
#     return [
#         "/Users/rt582/miniforge3/envs/cmipper/bin/python",
#         script_fp,
#     ] + arguments  # TODO: depersonalise this

import os
import numpy as np

import time
import warnings
import xarray as xa

import argparse
import requests
from pathlib import Path
import re

from cdo import Cdo

from coralshift.functions_creche import (
    lat_lon_string_from_tuples,
    gen_seafloor_indices,
    extract_seafloor_vals,
    generate_remapping_file,
    FileName,
)
from coralshift.processing.spatial_data import process_xa_d

import config

"""
Script to download monthly-averaged CMIP6 climate simulation runs from the Earth
System Grid Federation (ESFG):
https://esgf-node.llnl.gov/search/cmip6/.
The simulations are regridded from latitude/longitude...

The --source_id and --member_id command line inputs control which climate model
and model run to download.

The `download_dict` dictates which variables to download from each climate
model. Entries within the variable dictionaries of `variable_dict` provide
further specification for the variable to download - e.g. whether it is on an
ocean grid and whether to download data at a specified pressure level. All this
information is used to create the `query` dictionary that is passed to
utils.esgf_search to find download links for the variable. The script loops
through each variable, downloading those for which 'include' is True in the
variable dictionary.

Variable files are saved to relevant folders beneath cmip6/<source_id>/<member_id>/
directory.

See download_cmip6_data_in_parallel.sh to download and regrid multiple climate
simulations in parallel using this script.
"""

# COMMAND LINE INPUT
################################################################################
parser = argparse.ArgumentParser()

# model info
source_id = parser.add_argument("--source_id", default="EC-Earth3P-HR", type=str)
member_id = parser.add_argument("--member_id", default="r1i1p2f1", type=str)
variable_id = parser.add_argument("--variable_id", default="tos", type=str)
# spatial info
LATS = parser.add_argument("--lats", action="append", type=float)
LONS = parser.add_argument("--lons", action="append", type=float)
LEVS = parser.add_argument("--levs", action="append", type=int)
# TODO: get this from model
RESOLUTION = parser.add_argument("--resolution", default=0.25, type=float)
# processing info
do_download = parser.add_argument("--do_download", default=True, type=bool)
do_overwrite = parser.add_argument("--do_overwrite", default=False, type=bool)
do_delete_og = parser.add_argument("--do_delete_og", default=False, type=bool)
do_concat_by_time = parser.add_argument("--do_concat_by_time", default=True, type=bool)
do_merge_by_vars = parser.add_argument("--do_merge_by_vars", default=True, type=bool)
do_crop = parser.add_argument("--do_crop", default=True, type=bool)
do_regrid = parser.add_argument("--do_regrid", default=True, type=bool)

commandline_args = parser.parse_args()

# User download options
################################################################################
print("\nProcessing data for {}, {}\n".format(source_id, variable_id))

# model info
source_id = commandline_args.source_id
variable_id = commandline_args.variable_id
member_id = commandline_args.member_id
# spatial info


LATS = commandline_args.lats
LONS = commandline_args.lons
LEVS = commandline_args.levs
LEVS = [0, 20]
INT_LATS = [int(lat) for lat in LATS]
INT_LONS = [int(lon) for lon in LONS]
RESOLUTION = commandline_args.resolution
# processing info
do_download = commandline_args.do_download
do_overwrite = commandline_args.do_overwrite
do_delete_og = commandline_args.do_delete_og
do_concat_by_time = commandline_args.do_concat_by_time
do_merge_by_vars = commandline_args.do_merge_by_vars
do_crop = commandline_args.do_crop
do_regrid = commandline_args.do_regrid

if do_crop and not do_regrid:
    print("WARNING: cropping without regridding may lead to unexpected results")

# Orlando functions
################################################################################


# these variable names (lat, lon, lev) may be specific to EC-Earth3P-HR
def _spatial_crop(ds):
    spatial_ds = ds.sel(
        lat=slice(min(LATS), max(LATS)), lon=slice(min(LONS), max(LONS))
    )
    if "lev" in ds.coords:
        return spatial_ds.sel(lev=slice(min(LEVS), max(LEVS)))
    else:
        return spatial_ds


# Below taken from https://hub.binder.pangeo.io/user/pangeo-data-pan--cmip6-examples-ro965nih/lab
def esgf_search(
    server="https://esgf-node.llnl.gov/esg-search/search",
    files_type="OPENDAP",
    local_node=False,
    latest=True,
    project="CMIP6",
    verbose1=False,
    verbose2=False,
    format="application%2Fsolr%2Bjson",
    use_csrf=False,
    **search,
):
    client = requests.session()
    payload = search
    payload["project"] = project
    payload["type"] = "File"
    if latest:
        payload["latest"] = "true"
    if local_node:
        payload["distrib"] = "false"
    if use_csrf:
        client.get(server)
        if "csrftoken" in client.cookies:
            # Django 1.6 and up
            csrftoken = client.cookies["csrftoken"]
        else:
            # older versions
            csrftoken = client.cookies["csrf"]
        payload["csrfmiddlewaretoken"] = csrftoken

    payload["format"] = format

    offset = 0
    numFound = 10000
    all_files = []
    files_type = files_type.upper()
    while offset < numFound:
        payload["offset"] = offset
        url_keys = []
        for k in payload:
            url_keys += ["{}={}".format(k, payload[k])]

        url = "{}/?{}".format(server, "&".join(url_keys))
        if verbose1:
            print(url)
        r = client.get(url)
        r.raise_for_status()
        resp = r.json()["response"]
        numFound = int(resp["numFound"])
        resp = resp["docs"]
        offset += len(resp)
        for d in resp:
            if verbose2:
                for k in d:
                    print("{}: {}".format(k, d[k]))
            url = d["url"]
            for f in d["url"]:
                sp = f.split("|")
                if sp[-1] == files_type:
                    all_files.append(sp[0].split(".html")[0])
    return sorted(all_files)


# Download information
################################################################################

download_dict = {
    "EC-Earth3P-HR": {
        "experiment_ids": ["hist-1950"],
        "member_ids": ["r1i1p2f1"],
        "data_nodes": [
            # "cmip.bcc.cma.cn",  # listed in metadata from https://esgf-node.llnl.gov/search/cmip6/
            "esgf3.dkrz.de",
            "esgf-data1.llnl.gov",
            "esgf.ceda.ac.uk",
        ],
        "frequency": "mon",
        "variable_dict": {
            # "hfds": {   # Downward Heat Flux at Sea Water Surface
            #     "include": True,
            #     "table_id": "Omon",
            #     "plevels": None,    # surface variable. May automate this more e.g lookup dict of surface/levels var
            # },
            "rsdo": {  # Downwelling Shortwave Radiation in Sea Water
                "include": True,
                "table_id": "Omon",
                "plevels": [-1],
            },
            # "umo": {    # Ocean Mass X Transport
            #     "include": True,
            #     "table_id": "Omon",
            #     "plevels": None,
            # },
            # "vmo": {    # Ocean Mass Y Transport
            #     "include": True,
            #     "table_id": "Omon",
            #     "plevels": None,
            # },
            "mlotst": {  # Ocean Mixed Layer Thickness Defined by Sigma T
                "include": True,
                "table_id": "Omon",
                "plevels": [None],
            },
            "so": {  # Sea Water Salinity
                "include": True,
                "table_id": "Omon",
                "plevels": [-1],
            },
            "thetao": {  # Sea Water Potential Temperature
                "include": True,
                "table_id": "Omon",
                "plevels": [-1],
            },
            "uo": {  # Sea Water X Velocity
                "include": True,
                "table_id": "Omon",
                "plevels": [-1],
            },
            "vo": {  # Sea Water Y Velocity
                "include": True,
                "table_id": "Omon",
                "plevels": [-1],
            },
            # "wfo": {    # Water Flux into Sea Water
            #     "include": True,
            #     "table_id": "Omon",
            #     "plevels": None,
            # },
            "tos": {  # Sea Surface Temperature
                "include": True,
                "table_id": "Omon",  # also available at 3hr
                "plevels": [None],
            },
        },
    }
}


# TODO: remove testing folder
download_folder = os.path.join(
    Path(config.cmip6_data_folder), source_id, member_id, "testing"
)
if not os.path.exists(download_folder):
    os.makedirs(download_folder)

# Download
################################################################################

# Ignore "Missing CF-netCDF variable" warnings from download
warnings.simplefilter("ignore", UserWarning)


def download_cmip_data(
    download_dict: dict[str, dict[str, dict[str, str]]],
    download_folder: str,
    do_download: bool,
    do_regrid: bool,
    do_delete_og: bool,
    do_crop: bool,
):
    source_id_dict = download_dict[source_id]
    variable_dict = source_id_dict["variable_dict"]
    variable_id_dict = variable_dict[variable_id]

    tic = time.time()

    for member_id in source_id_dict["member_ids"]:
        # DOWNLOAD
        if do_download:
            if variable_id_dict["include"] is False:
                continue
            # BUILD QUERY
            query = {
                "source_id": source_id,
                "member_id": member_id,
                "frequency": source_id_dict["frequency"],
                "variable_id": variable_id,
                "table_id": variable_id_dict["table_id"],
            }

            if (
                "ocean_variable" in variable_id_dict.keys()
            ):  # this originally included "or EC-Earth3" (lower-res version)
                # TODO: figure out what this is doing. May be redundant
                query["grid_label"] = "gr"
            else:
                query["grid_label"] = "gn"

            print("\n\n{}: ".format(variable_id), end="", flush=True)
            print("searching ESGF... ", end="", flush=True)
            results = []

            for experiment_id in source_id_dict["experiment_ids"]:
                query["experiment_id"] = experiment_id

                experiment_id_results = []
                for data_node in source_id_dict["data_nodes"]:
                    query["data_node"] = data_node
                    # EXECUTE QUERY
                    experiment_id_results.extend(esgf_search(**query))

                    # Keep looping over possible data nodes until the experiment data is found
                    if len(experiment_id_results) > 0:
                        print("found {}, ".format(experiment_id), end="", flush=True)
                        results.extend(experiment_id_results)
                        break  # Break out of the loop over data nodes when data found

                results = list(set(results))
                print("found {} files. ".format(len(results)), end="", flush=True)

            # SET UP FILE PATHS REFERENCING
            fpaths_og = {}
            fpaths_regridded = {}

            plevels = (
                list(variable_id_dict["plevels"])
                if not isinstance(variable_id_dict["plevels"], list)
                else variable_id_dict["plevels"]
            )
            for plevel in plevels:
                # reset indices for each level
                seafloor_indices = None
                failed_regrids = []

                print("downloading individual files... ", flush=True)
                ind_download_folder = os.path.join(download_folder, variable_id)
                if not os.path.exists(ind_download_folder):
                    os.makedirs(ind_download_folder)

                for i, result in enumerate(results):
                    date_range = result.split("_")[-1].split(".")[0]

                    fname_og = FileName(
                        variable_id=variable_id,
                        grid_type="tripolar",  # TODO: get this info from the associated dataset
                        fname_type="individual",
                        levs=LEVS,
                        date_range=date_range,
                        plevels=plevel,
                    ).construct_fname()

                    print(
                        f"\n{i}: handling {fname_og}",
                        flush=True,
                    )

                    save_fp = os.path.join(ind_download_folder, fname_og)

                    if os.path.exists(save_fp):
                        print(
                            f"\t{i}: skipping download due to existing file: {save_fp}",
                            flush=True,
                        )
                        fpaths_og[i] = save_fp
                    else:
                        # OPEN AND SELECT CORRECT PRESSURE LEVELS
                        if not plevel:  # if surface variable: chunk by time
                            ds = xa.open_dataset(
                                result, decode_times=True, chunks={"time": "499MB"}
                            )[variable_id]
                        else:  # if seafloor or specified pressure, open with limited levels, chunked spatially
                            print("\n\n\n\nmin(LEVS):", type((LEVS)), type(min(LEVS)))
                            ds = xa.open_dataset(
                                result,
                                decode_times=True,  # not all times easily decoded
                                chunks={"i": 200, "j": 200},
                            ).isel(lev=slice(min(LEVS), max(LEVS)))
                            if plevel == -1:  # if seafloor
                                if (
                                    seafloor_indices is None
                                ):  # if seafloor indices not yet calculated, calculate
                                    print(
                                        "\tdetermining seafloor indices... ", flush=True
                                    )
                                    seafloor_indices = gen_seafloor_indices(
                                        ds.isel(time=0),
                                        var=variable_id,
                                    )
                                    seafloor_indices = np.broadcast_to(
                                        seafloor_indices,
                                        (
                                            len(ds.time),
                                            len(ds.j),
                                            len(ds.i),
                                        ),  # these variable names may differ by model
                                    )

                                print("\textracting seafloor values...", flush=True)
                                cmip6_array = extract_seafloor_vals(
                                    ds[variable_id], seafloor_indices
                                )
                                ds[variable_id] = (["time", "j", "i"], cmip6_array)
                            else:  # if surface
                                print(
                                    f"extracting {plevel / 100:.03f} hPa",
                                    flush=True,
                                )
                                ds = ds.sel(lev=plevel)[variable_id]

                        print(
                            f"\t{i}: saving {fname_og} file: {save_fp}",
                            flush=True,
                        )
                        fpaths_og[i] = save_fp
                        # TODO: change dtype?
                        ds.to_netcdf(save_fp)

                    if do_regrid:
                        regrid_dir_fp = os.path.join(
                            download_folder, "regridded", variable_id
                        )
                        if not os.path.exists(regrid_dir_fp):
                            os.makedirs(regrid_dir_fp)
                        remap_template_fp = (
                            Path(config.cmip6_data_folder)
                            / source_id
                            / f"{source_id}_remap_template.txt"
                        )
                        if not Path(remap_template_fp).exists():
                            generate_remapping_file(
                                ds,
                                resolution=RESOLUTION,
                                out_grid="latlon",
                                remap_template_fp=remap_template_fp,
                            )
                        # initialise instance of Cdo for regridding
                        cdo = Cdo()

                        fname_regrid = FileName(
                            variable_id=variable_id,
                            grid_type="latlon",
                            fname_type="individual",
                            plevels=plevel,
                            levs=LEVS,
                            date_range=date_range,
                        ).construct_fname()
                        # add to list of regridded files
                        fpaths_regridded[i] = os.path.join(regrid_dir_fp, fname_regrid)

                        if not os.path.exists(fpaths_regridded[i]):
                            print(f"\t{i}: regridding to {fpaths_regridded[i]}...")
                            try:
                                cdo.remapbil(  # TODO: different types of regridding. Will have to update filenames
                                    remap_template_fp,
                                    input=fpaths_og[i],
                                    output=fpaths_regridded[i],
                                )
                            except:  # noqa # TODO: find proper exception
                                print(
                                    f"regridding failed for {fpaths_og[i]}, skipping...",
                                    flush=True,
                                )
                                failed_regrids.append(fpaths_og[i])
                                continue
                        else:
                            print(
                                f"\t{i}: skipping regrid due to existing file: {fpaths_regridded[i]}",
                                flush=True,
                            )
                            # TODO: process_xa_d on these files then re-waving

                        if do_delete_og:
                            os.remove(fpaths_og[i])

                        ds = xa.open_dataset(fpaths_regridded[i])

                    if do_crop:
                        cropped_dir_name = f"cropped_{lat_lon_string_from_tuples(INT_LATS, INT_LONS).upper()}"
                        cropped_dir_fp = os.path.join(
                            download_folder, "regridded", cropped_dir_name, variable_id
                        )
                        if not os.path.exists(cropped_dir_fp):
                            os.makedirs(cropped_dir_fp)

                        fname_cropped = FileName(
                            variable_id=variable_id,
                            grid_type="latlon",
                            fname_type="individual",
                            lats=INT_LATS,
                            lons=INT_LONS,
                            levs=LEVS,
                            plevels=plevel,
                            date_range=date_range,
                        ).construct_fname()

                        cropped_save_fp = Path(cropped_dir_fp) / fname_cropped
                        if not os.path.exists(cropped_save_fp):
                            ds = process_xa_d(_spatial_crop(ds))
                            print(
                                f"\t{i}: saving {fname_cropped} file: {cropped_save_fp}",
                                flush=True,
                            )
                            # TODO: change dtype?
                            ds.to_netcdf(cropped_save_fp)
                        else:
                            print(
                                f"\t{i}: skipping cropping due to existing file: {cropped_save_fp}",
                                flush=True,
                            )

            download_tic = time.time() - tic
            actions = [
                action
                for action, flag in [("regridding", do_regrid), ("cropping", do_crop)]
                if flag
            ]
            message = f"Downloading/{'/'.join(actions)}" if actions else "Downloading"
            print(
                f"\n{message} took {np.floor(download_tic / 60):.0f}m:{download_tic % 60:.0f}s."
            )

            print(
                f"\n{len(failed_regrids)} regrids failed.\n"
                if len(failed_regrids) == 0
                else f"\n{len(failed_regrids)} regrids failed. The following are likely corrupted: \n{failed_regrids}\n"
            )

    # TODO: put the concatenation by time and by variable into separate functions?


def concat_cmip_files_by_time(download_folder, source_id_dict, do_crop):
    # CONCATENATE BY TIME
    tic = time.time()

    conc_var_dir = os.path.join(download_folder, "concatted_vars")
    if do_crop:
        conc_var_dir += f"_{lat_lon_string_from_tuples(INT_LATS, INT_LONS).upper()}"
    if not os.path.exists(conc_var_dir):
        os.makedirs(conc_var_dir)

    # fetch variable_id to fetch all files to be concatted
    for variable_id in list(source_id_dict["variable_dict"].keys()):
        if do_crop:
            variable_dir = (
                Path(download_folder)
                / "regridded"
                / f"cropped_{lat_lon_string_from_tuples(INT_LATS, INT_LONS).upper()}"
                / variable_id
            )
        else:
            # directory with individual files for single variable
            variable_dir = Path(download_folder) / "regridded" / variable_id

        if not os.path.exists(variable_dir):
            print(f"{variable_dir} does not exist, skipping", flush=True)
            continue

        fps = list(variable_dir.glob("*.nc"))
        oldest_file = min(
            fps, key=lambda filename: int(re.findall(r"\d{4}", str(filename))[0])
        )
        newest_file = max(
            fps, key=lambda filename: int(re.findall(r"\d{4}", str(filename))[0])
        )
        oldest_date = str(oldest_file.name).split("_")[-1].split("-")[0]
        newest_date = str(newest_file.name).split("_")[-1].split("-")[1].split(".")[0]

        fname = FileName(
            variable_id=variable_id,
            grid_type="latlon",
            fname_type="time_concatted",
            lats=LATS,
            lons=LONS,
            levs=LEVS,
            plevels=source_id_dict["variable_dict"][variable_id]["plevels"],
            date_range=[oldest_date, newest_date],
        ).construct_fname()
        concatted_fp = os.path.join(conc_var_dir, fname)

        if os.path.exists(concatted_fp):
            print(f"concatenated file already exists at {concatted_fp}", flush=True)
        else:
            print(f"concatenating {variable_id} files by time... ", flush=True)
            nc_fps = list(Path(variable_dir).glob("*.nc"))
            concatted = xa.open_mfdataset(nc_fps)

            # decode time
            concatted = concatted.convert_calendar(
                "gregorian", dim="time"
            )  # may not be universal for all models
            print(
                f"saving concatenated file to {concatted_fp}... ",
                flush=True,
            )
            concatted.to_netcdf(concatted_fp)

    time_concat_tic = time.time() - tic
    print(
        f"\nConcatenating files by time took {np.floor(time_concat_tic / 60):.0f}m:{time_concat_tic % 60:.0f}s.\n"
    )


def merge_cmip_data_by_variables(download_folder, do_crop: bool):
    tic = time.time()

    # MERGE VARIABLES
    conc_var_dir = os.path.join(download_folder, "concatted_vars")
    if do_crop:
        conc_var_dir += f"_{lat_lon_string_from_tuples(INT_LATS, INT_LONS).upper()}"

    assert os.path.exists(conc_var_dir) and os.path.isdir(
        conc_var_dir
    ), f"{conc_var_dir} does not exist or is not a directory. Make sure do_concat_by_time is 'True'."
    var_nc_fps = list(Path(conc_var_dir).glob("*.nc"))

    oldest_file = min(
        var_nc_fps, key=lambda filename: int(re.findall(r"\d{4}", str(filename))[0])
    )
    newest_file = max(
        var_nc_fps, key=lambda filename: int(re.findall(r"\d{4}", str(filename))[0])
    )
    oldest_date = str(oldest_file.name).split("_")[-1].split("-")[0]
    newest_date = str(newest_file.name).split("_")[-1].split("-")[1].split(".")[0]

    vars = [str(fname.name).split("_")[0] for fname in var_nc_fps]
    # sort vars in alphabetical for consistency between files
    vars.sort()
    # TODO: sort out naming

    merged_fname = FileName(
        # variable_id=list(variable_dict.keys()),
        variable_id=vars,
        grid_type="latlon",
        fname_type="var_concatted",
        lats=LATS,
        lons=LONS,
        levs=LEVS,
        plevels=LEVS,
        date_range=[oldest_date, newest_date],
    ).construct_fname()

    print("var_nc_fps", var_nc_fps)

    merged_fp = os.path.join(download_folder, merged_fname)
    print("merged_fp", merged_fp)
    if not os.path.exists(merged_fp):
        print(
            f"\nmerging variable files and saving to {merged_fp}... ",
            flush=True,
        )
        dss = [xa.open_dataset(fp) for fp in var_nc_fps]
        merged = xa.merge(dss)
        merged.to_netcdf(merged_fp)
    else:
        print(
            f"\nmerged file already exists at {merged_fp}",
            flush=True,
        )

    var_concat_tic = time.time() - tic
    print(
        f"\nMerging files by variable took {np.floor(var_concat_tic / 60):.0f}m:{var_concat_tic % 60:.0f}s."
    )


def main():
    tic = time.time()
    print(f"TIME CREATED: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(tic))}")

    download_cmip_data(
        download_dict,
        download_folder,
        do_download,
        do_regrid,
        do_delete_og,
        do_crop,
    )

    if do_concat_by_time:
        concat_cmip_files_by_time(download_folder, download_dict[source_id], do_crop)

    if do_merge_by_vars:
        merge_cmip_data_by_variables(download_folder, do_crop)

    dur = time.time() - tic
    print(f"\n\nTOTAL DURATION: {np.floor(dur / 60):.0f}m:{dur % 60:.0f}s\n")


main()

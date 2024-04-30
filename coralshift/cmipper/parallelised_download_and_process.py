import os
import numpy as np

import time
import warnings
import xarray as xa
import argparse
from tqdm.auto import tqdm

from pathlib import Path
import re

from cdo import Cdo

from cmipper import config, utils, downloading, file_ops

"""
Script to download monthly-averaged CMIP6 climate simulation runs from the Earth
System Grid Federation (ESFG):
https://esgf-node.llnl.gov/search/cmip6/.
The simulations are regridded from latitude/longitude...

The --source_id and --member_id command line inputs control which climate model
and model run to download.

The `model_dict` dictates which variables to download from each climate
model. Entries within the variable dictionaries of `variable_dict` provide
further specification for the variable to download - e.g. whether it is on an
ocean grid and whether to download data at a specified pressure level. All this
information is used to create the `query` dictionary that is passed to
downloading.esgf_search to find download links for the variable. The script loops
through and downloads each variable specified in the variable dictionary.

Variable files are saved to relevant folders beneath cmip6/<source_id>/<member_id>/
directory.

See download_cmip6_data_in_parallel.sh to download and regrid multiple climate
simulations in parallel using this script.
# TODO: add notebook option
"""


# Ignore "Missing CF-netCDF variable" warnings from download
warnings.simplefilter("ignore", UserWarning)


# Download function
################################################################################


def download_cmip_variable_data(
    source_id: str,
    member_id: str,
    variable_id: str,
):
    """
    This downloads a single variable, for however many experiments are specified in the source_id_dict.
    """
    # read download values
    source_id_dict = utils.read_yaml(config.model_info)[source_id]
    download_config_dict = utils.read_yaml(config.download_config)

    # TODO: parallelise by source_id and member_id
    variable_id_dict = source_id_dict["variable_dict"][variable_id]

    # spatial values
    LATS = sorted(download_config_dict["lats"])
    LONS = sorted(download_config_dict["lons"])
    INT_LATS = [int(lat) for lat in LATS]
    INT_LONS = [int(lon) for lon in LONS]
    LEVS = sorted([abs(val) for val in download_config_dict["levs"]])
    RESOLUTION = source_id_dict["resolution"]
    # processing values
    do_regrid = download_config_dict["processing"]["do_regrid"]
    do_delete_og = download_config_dict["processing"]["do_delete_og"]
    do_crop = download_config_dict["processing"]["do_crop"]

    if do_crop and not do_regrid:
        print("WARNING: cropping without regridding may lead to unexpected results")

    # file setting
    download_dir = (
        config.cmip6_data_dir / source_id / member_id
    )  # TODO: remove testing folder
    if not download_dir.exists():
        download_dir.mkdir(parents=True, exist_ok=True)

    tic = time.time()

    print(f"TIME CREATED: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(tic))}")
    print(f"\nProcessing data for {source_id}, {member_id}, {variable_id}\n")

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
    print("searching ESGF servers... \n", end="", flush=True)
    results = []

    for experiment_id in source_id_dict["experiment_ids"]:
        query["experiment_id"] = experiment_id

        experiment_id_results = []
        for data_node in source_id_dict["data_nodes"]:
            query["data_node"] = data_node
            # EXECUTE QUERY
            experiment_id_results.extend(downloading.esgf_search(**query))

            # Keep looping over possible data nodes until the experiment data is found
            if len(experiment_id_results) > 0:
                print("\nfound {}, ".format(experiment_id), end="", flush=True)
                results.extend(experiment_id_results)
                break  # Break out of the loop over data nodes when data found

        results = list(set(results))  # remove any duplicate values

        YEAR_RANGE = sorted(download_config_dict["experiment_ids"][experiment_id])
        # skip any unrequired dates
        relevant_results = file_ops.find_files_for_time(results, year_range=YEAR_RANGE)
        print(
            f"""found {len(results)} files on {data_node} node of which {len(relevant_results)}
            fall(s) within required date range.""".format(
                end="", flush=True
            )
        )

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
            ind_download_dir = download_dir / "og_grid" / variable_id
            if not ind_download_dir.exists():
                ind_download_dir.mkdir(parents=True, exist_ok=True)

            for i, result in enumerate(relevant_results):
                date_range = result.split("_")[-1].split(".")[0]

                fname_og = file_ops.FileName(
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

                save_fp = ind_download_dir / fname_og

                if save_fp.exists():
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
                        ds = xa.open_dataset(
                            result,
                            decode_times=True,  # not all times easily decoded
                            chunks={"i": "499MB"},
                        ).isel(lev=slice(min(LEVS), max(LEVS)))
                        if plevel == -1:  # if seafloor
                            if (
                                seafloor_indices is None
                            ):  # if seafloor indices not yet calculated, calculate
                                print("\tdetermining seafloor indices... ", flush=True)
                                seafloor_indices = utils.gen_seafloor_indices(
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

                            print("\textracting seafloor values...\n", flush=True)
                            cmip6_array = utils.extract_seafloor_vals(
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
                    regrid_dir_fp = download_dir / "regridded" / variable_id
                    if not regrid_dir_fp.exists():
                        regrid_dir_fp.mkdir(parents=True, exist_ok=True)
                    remap_template_fp = (
                        config.cmip6_data_dir
                        / source_id
                        / f"{source_id}_remap_template.txt"
                    )
                    if not remap_template_fp.exists():
                        utils.generate_remapping_file(
                            ds,
                            remap_template_fp=remap_template_fp,
                            resolution=RESOLUTION,
                            out_grid="latlon",
                        )
                    # initialise instance of Cdo for regridding
                    cdo = Cdo()

                    fname_regrid = file_ops.FileName(
                        variable_id=variable_id,
                        grid_type="latlon",
                        fname_type="individual",
                        plevels=plevel,
                        levs=LEVS,
                        date_range=date_range,
                    ).construct_fname()
                    # add to list of regridded files
                    fpaths_regridded[i] = regrid_dir_fp / fname_regrid

                    if not fpaths_regridded[i].exists():
                        print(f"\t{i}: regridding to {fpaths_regridded[i]}...")
                        try:
                            cdo.remapbil(  # TODO: different types of regridding. Will have to update filenames
                                str(remap_template_fp),
                                input=str(fpaths_og[i]),
                                output=str(fpaths_regridded[i]),
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
                        print(fpaths_og[i])
                        os.remove(fpaths_og[i])

                    try:
                        ds = xa.open_dataset(fpaths_regridded[i])
                    except Exception as e:
                        print(
                            f"\nregridded file {fpaths_regridded[i]} appears corrupted, skipping...",
                            flush=True,
                        )
                        print("Error: \n", e, flush=True)
                        failed_regrids.append(fpaths_regridded[i])
                        continue

                if do_crop:
                    cropped_dir_name = f"cropped_{utils.lat_lon_string_from_tuples(INT_LATS, INT_LONS).upper()}"
                    cropped_dir_fp = (
                        download_dir / "regridded" / cropped_dir_name / variable_id
                    )
                    if not cropped_dir_fp.exists():
                        cropped_dir_fp.mkdir(parents=True, exist_ok=True)

                    fname_cropped = file_ops.FileName(
                        variable_id=variable_id,
                        grid_type="latlon",
                        fname_type="individual",
                        lats=INT_LATS,
                        lons=INT_LONS,
                        levs=LEVS,
                        plevels=plevel,
                        date_range=date_range,
                    ).construct_fname()

                    cropped_save_fp = cropped_dir_fp / fname_cropped
                    if not cropped_save_fp.exists():
                        # N.B. may be different between models, and may need to include levs
                        ds = utils.process_xa_d(ds).sel(
                            latitude=slice(min(LATS), max(LATS)),
                            longitude=slice(min(LONS), max(LONS)),
                        )
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
        actions_undertaken = [
            action
            for action, flag in [("regridding", do_regrid), ("cropping", do_crop)]
            if flag
        ]
        message = (
            f"Downloading/{'/'.join(actions_undertaken)}"
            if actions_undertaken
            else "Downloading"
        )
        print(
            f"\n{message} took {np.floor(download_tic / 60):.0f}m:{download_tic % 60:.0f}s."
        )

        print(
            f"\n{len(failed_regrids)} regrid(s) failed.\n"
            if len(failed_regrids) == 0
            else f"\n{len(failed_regrids)} regrid(s) failed. The following are/is likely corrupted: \n"
            + "\n".join(str(item) for item in list(set(failed_regrids)))
            + "\n"
        )


# Processing (concatenation by time, merging by variables) functions
################################################################################


def concat_cmip_files_by_time(source_id, experiment_id, member_id, variable_id):
    download_dir = (
        config.cmip6_data_dir / source_id / member_id / "newtest"
    )  # TODO: remove testing folder
    source_id_dict = utils.read_yaml(config.model_info)[source_id]
    download_config_dict = utils.read_yaml(config.download_config)

    DO_CROP = download_config_dict["processing"]["do_crop"]
    LATS = sorted(download_config_dict["lats"])
    LONS = sorted(download_config_dict["lons"])
    INT_LATS = [int(lat) for lat in LATS]
    INT_LONS = [int(lon) for lon in LONS]
    LEVS = sorted([abs(val) for val in download_config_dict["levs"]])
    YEAR_RANGE = download_config_dict["experiment_ids"][experiment_id]

    # CONCATENATE BY TIME
    tic = time.time()

    conc_var_dir = download_dir / "regridded" / "concatted_vars"
    if DO_CROP:
        conc_var_dir = Path(
            str(conc_var_dir)
            + f"_{utils.lat_lon_string_from_tuples(INT_LATS, INT_LONS).upper()}"
        )
    if not conc_var_dir.exists():
        conc_var_dir.mkdir(parents=True, exist_ok=True)

    # fetch variable_id to fetch all files to be concatted
    # for variable_id in list(source_id_dict["variable_dict"].keys()):
    if DO_CROP:
        variable_dir = (
            download_dir
            / "regridded"
            / f"cropped_{utils.lat_lon_string_from_tuples(INT_LATS, INT_LONS).upper()}"
            / variable_id
        )
    else:
        # directory with individual files for single variable
        variable_dir = download_dir / "regridded" / variable_id

    # if not variable_dir.exists():
    #     print(f"{variable_dir} does not exist, skipping", flush=True)
    #     continue

    fps = list(variable_dir.glob("*.nc"))

    if YEAR_RANGE:
        oldest_date = str(min(YEAR_RANGE)) + "00"
        newest_date = str(max(YEAR_RANGE) - 1) + "12"
    else:
        oldest_file = min(
            fps, key=lambda filename: int(re.findall(r"\d{4}", str(filename))[0])
        )
        newest_file = max(
            fps, key=lambda filename: int(re.findall(r"\d{4}", str(filename))[0])
        )
        oldest_date = str(oldest_file.name).split("_")[-1].split("-")[0]
        newest_date = str(newest_file.name).split("_")[-1].split("-")[1].split(".")[0]

    fname = file_ops.FileName(
        variable_id=variable_id,
        grid_type="latlon",
        fname_type="time_concatted",
        lats=LATS,
        lons=LONS,
        levs=LEVS,
        plevels=source_id_dict["variable_dict"][variable_id]["plevels"],
        date_range=[oldest_date, newest_date],
    ).construct_fname()
    concatted_fp = conc_var_dir / fname

    if concatted_fp.exists():
        print(f"\nconcatenated file already exists at {str(concatted_fp)}", flush=True)
    else:
        # fetch all the filepaths of the files containing dates between oldest_date and newest_date
        nc_fps = [
            fp
            for fp in fps
            if oldest_date <= str(fp.name).split("_")[-1].split("-")[0]
            and newest_date >= str(fp.name).split("_")[-1].split("-")[1].split(".")[0]
        ]

        if len(nc_fps) == 0:
            print(
                f"skipping {variable_id} since no files found between {oldest_date} and {newest_date}..."
            )
            # continue
        else:

            print(f"concatenating {variable_id} files by time... ", flush=True)

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


def merge_cmip_data_by_variables(source_id, experiment_id, member_id):
    download_dir = (
        config.cmip6_data_dir / source_id / member_id / "newtest" / "regridded"
    )  # TODO: remove testing folder
    # source_id_dict = utils.read_yaml(config.model_info)[source_id]
    download_config_dict = utils.read_yaml(config.download_config)

    DO_CROP = download_config_dict["processing"]["do_crop"]
    LATS = sorted(download_config_dict["lats"])
    LONS = sorted(download_config_dict["lons"])
    INT_LATS = [int(lat) for lat in LATS]
    INT_LONS = [int(lon) for lon in LONS]
    LEVS = sorted([abs(val) for val in download_config_dict["levs"]])
    YEAR_RANGE = download_config_dict["experiment_ids"][experiment_id]

    tic = time.time()

    # MERGE VARIABLES
    conc_var_dir = download_dir / "concatted_vars"
    if DO_CROP:
        conc_var_dir = str(conc_var_dir) + (
            f"_{utils.lat_lon_string_from_tuples(INT_LATS, INT_LONS).upper()}"
        )

    if not Path(conc_var_dir).exists():
        conc_var_dir.mkdir(parents=True, exist_ok=True)

    # select all files in conc_var_dir which have correct YEAR_RANGE
    oldest_date = str(min(YEAR_RANGE)) + "00"
    newest_date = str(max(YEAR_RANGE) - 1) + "12"
    YEAR_RANGE_str = f"{oldest_date}-{newest_date}"
    var_nc_fps = list(Path(conc_var_dir).glob(f"*{YEAR_RANGE_str}.nc"))

    vars = [str(fname.name).split("_")[0] for fname in var_nc_fps]
    # sort vars in alphabetical for consistency between files
    vars.sort()

    merged_fname = file_ops.FileName(
        variable_id=vars,
        grid_type="latlon",
        fname_type="var_concatted",
        lats=LATS,
        lons=LONS,
        levs=LEVS,
        plevels=LEVS,  # TODO: Hmm? What's going on here?
        date_range=[oldest_date, newest_date],
    ).construct_fname()

    merged_fp = download_dir / merged_fname
    if not merged_fp.exists():
        print(
            f"\nmerging variable files and saving to {merged_fp}... ",
            flush=True,
        )
        dss = [xa.open_dataset(fp) for fp in var_nc_fps]
        merged = xa.merge(dss)
        merged.to_netcdf(merged_fp)

        var_concat_tic = time.time() - tic
        print(
            f"\nMerging files by variable took {np.floor(var_concat_tic / 60):.0f}m:{var_concat_tic % 60:.0f}s."
        )
    else:
        print(
            f"\nmerged file already exists at {merged_fp}",
            flush=True,
        )


def delete_corrupt_files(source_id, member_id):
    """Some files may fail to regrid, usually due to partial download. This function attempts to remap
    the file to some arbitrary path and deletes any files which fail.

    TODO: any way to speed this up?
    TODO: should also (or another function) attempt to open final files and check if errors are thrown
    """
    download_dir = config.cmip6_data_dir / source_id / member_id / "newtest"
    og_grid_dir = download_dir / "og_grid" / "umo"
    output_dir = og_grid_dir / "temp_regrid"
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    var_dirs = [entry for entry in og_grid_dir.iterdir() if entry.is_dir()]
    var_dirs = [og_grid_dir]

    corrupt = []
    legit = []
    for variable_dir in var_dirs:
        nc_fps = list(variable_dir.glob("*.nc"))
        for nc_fp in tqdm(nc_fps):
            cdo = Cdo()
            try:
                output_fp = output_dir / nc_fp.name
                print(f"attempting remap of {nc_fp}...")
                cdo.remapbil(
                    str(
                        "/maps/rt582/cmipper/data/env_vars/cmip6/EC-Earth3P-HR/EC-Earth3P-HR_remap_template.txt"
                    ),
                    input=str(nc_fp),
                    output=str(output_fp),
                )
                # if successful remap, delete output file
                print(f"written successfully to {output_fp}")
                print("will now remove this")
                legit.append(nc_fp)
                os.remove(output_fp)
            except Exception:
                print(f"remap failed for {nc_fp}, to deleting...")
                corrupt.append(nc_fp)
                # os.remove(nc_fp)
    os.remove(output_dir)


def process_cmip6_data(source_id, experiment_id, member_id, variable_id):
    tic = time.time()
    print(f"TIME CREATED: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(tic))}")

    concat_cmip_files_by_time(source_id, experiment_id, member_id, variable_id)
    # merge_cmip_data_by_variables(source_id, experiment_id, member_id)

    dur = time.time() - tic
    print(f"\n\nTOTAL DURATION: {np.floor(dur / 60):.0f}m:{dur % 60:.0f}s\n")


def main():

    # COMMAND LINE INPUT FROM BASH SCRIPT
    ################################################################################
    parser = argparse.ArgumentParser()

    # model info
    source_id = parser.add_argument("--source_id", default="EC-Earth3P-HR", type=str)
    member_id = parser.add_argument("--member_id", default="r1i1p2f1", type=str)
    variable_id = parser.add_argument("--variable_id", default="tos", type=str)
    experiment_id = parser.add_argument(
        "--experiment_id", default="hist-1950", type=str
    )
    command = parser.add_argument("--command", default="download", type=str)

    commandline_args = parser.parse_args()

    source_id = commandline_args.source_id
    member_id = commandline_args.member_id
    variable_id = commandline_args.variable_id
    experiment_id = commandline_args.experiment_id
    command = commandline_args.command
    # command = "download"
    # source_id = "EC-Earth3P-HR"
    # member_id = "r1i1p2f1"
    # variable_id = "tos"

    # DOWNLOAD DATA
    ################################################################################
    if command == "download":
        download_cmip_variable_data(source_id, member_id, variable_id)
    elif command == "delete_corrupt_files":
        print("delete_corrupt_files still to be checked")
    elif command == "process":
        process_cmip6_data(source_id, experiment_id, member_id, variable_id)
        # delete_corrupt_files(source_id, member_id)
        # delete_corrupt_files(source_id, member_id)


if __name__ == "__main__":
    main()

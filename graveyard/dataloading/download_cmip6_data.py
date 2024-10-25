# web communication
import argparse
import requests
from cdo import Cdo

# file handling
import config  # global file system paths. Could draw more from utils.directories script. One guarantees existence, but is lonnger/less clear. TBC.
import time
import warnings
from pathlib import Path
import os

import xarray as xa
import numpy as np

# from coralshift.utils import tuples_to_string # can't find coralshift module


# TODO: adapt this for different types of file (e.g. with depth dimension, regridding necessary etc.)
# TODO: slim down to necessary steps
# TODO: parallelise per variable
# TODO: put user download options into bash script

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

#### COMMAND LINE INPUT
# --------------------------------------------------------------------

parser = argparse.ArgumentParser()

# TODO: make more concise
parser.add_argument("--source_id", default="EC-Earth3P-HR", type=str)
parser.add_argument("--do_download", default=True, type=bool)
parser.add_argument("--do_overwrite", default=False, type=bool)
parser.add_argument("--do_delete_tripolar", default=False, type=bool)
parser.add_argument("--do_download_ind", default=True, type=bool)
parser.add_argument("--do_concat_by_time", default=False, type=bool)
parser.add_argument("--do_merge_by_vars", default=False, type=bool)
parser.add_argument("--do_crop", default=True, type=bool)
parser.add_argument("--do_regrid", default=True, type=bool)

commandline_args = parser.parse_args()

source_id = commandline_args.source_id
member_id = commandline_args.member_id
do_download = commandline_args.do_download
do_overwrite = commandline_args.do_overwrite
do_delete_tripolar = commandline_args.do_delete_tripolar
do_download_ind = commandline_args.do_download_ind
do_concat_by_time = commandline_args.do_concat_by_time
do_merge_by_vars = commandline_args.do_merge_by_vars
do_crop = commandline_args.do_crop
do_regrid = commandline_args.do_regrid

print("\n\nDownloading data for {}, {}\n".format(source_id, member_id))

####### User download options
################################################################################

# global for now
lats = [-40, 0]
lons = [130, 170]
levs = [0, 6]

# deprecated
# delete_latlon_data = True  # Delete lat-lon intermediate files is use_xarray is True
# compress = False


####### Orlando functions
################################################################################


if source_id == "EC-Earth3P-HR":

    def _spatial_crop(ds):
        """
        Cropping function for EC-Earth3P-HR model data. Speeds up processing to crop early
        """
        spatial_ds = ds.sel(
            lat=slice(min(lats), max(lats)), lon=slice(min(lons), max(lons))
        )
        if "lev" in ds.coords:
            return spatial_ds.sel(lev=slice(min(levs), max(levs)))
        else:
            return spatial_ds

else:
    # I don't see a diffference?
    def _spatial_crop(ds):
        spatial_ds = ds.sel(
            lat=slice(min(lats), max(lats)), lon=slice(min(lons), max(lons))
        )
        if "lev" in ds.coords:
            return spatial_ds.sel(lev=slice(min(levs), max(levs)))
        else:
            return spatial_ds


def gen_seafloor_indices(xa_da: xa.DataArray, var: str, dim: str = "lev"):
    """Generate indices of seafloor values for a given variable in an xarray dataset.

    Args:
        xa_da (xa.DataArray): xarray dataset containing variable of interest
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
    T, J, I = np.ogrid[:t, :j, :i]
    # select values from vals_array using indices_array
    return vals_array[T, indices_array, J, I]


def tuples_to_string(lats, lons):
    # Round the values in the tuples to the nearest integers
    round_lats = [round(lat) for lat in lats]
    round_lons = [round(lon) for lon in lons]

    # Create the joined string
    return f"lats_{min(round_lats)}-{max(round_lats)}_lons_{min(round_lons)}-{max(round_lons)}"


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


####### Download information
################################################################################

download_dict = {
    "EC-Earth3P-HR": {
        "experiment_ids": ["hist-1950"],
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
            #     "table_id": "Omon",  ###
            #     "plevels": None,    # surface variable. May automate this a little more e.g lookup dict of surface/levels variable
            # },
            "rsdo": {  # Downwelling Shortwave Radiation in Sea Water
                "include": True,
                "table_id": "Omon",  ###
                "plevels": [-1],  # -1 indicates lowest level (seafloor)
            },
            # "umo": {    # Ocean Mass X Transport
            #     "include": True,
            #     "table_id": "Omon",  ###
            #     "plevels": None,
            # },
            # "vmo": {    # Ocean Mass Y Transport
            #     "include": True,
            #     "table_id": "Omon",  ###
            #     "plevels": None,
            # },
            "mlotst": {  # Ocean Mixed Layer Thickness Defined by Sigma T
                "include": True,
                "table_id": "Omon",  ###
                "plevels": [-1],
            },
            "so": {  # Sea Water Salinity
                "include": True,
                "table_id": "Omon",  ###
                "plevels": [-1],
            },
            "thetao": {  # Sea Water Potential Temperature
                "include": True,
                "table_id": "Omon",  ###
                "plevels": [-1],
            },
            "uo": {  # Sea Water X Velocity
                "include": True,
                "table_id": "Omon",  ###
                "plevels": [-1],
            },
            "vo": {  # Sea Water Y Velocity
                "include": True,
                "table_id": "Omon",  ###
                "plevels": [-1],
            },
            # "wfo": {    # Water Flux into Sea Water
            #     "include": True,
            #     "table_id": "Omon",  ###
            #     "plevels": None,
            # },
            "tos": {  # Sea Surface Temperature
                "include": True,
                "table_id": "Omon",  ### also available at 3hr
                "plevels": None,
            },
        },
    }
}

download_folder = os.path.join(
    config.cmip6_data_folder, source_id, member_id, "testing"
)
if not os.path.exists(download_folder):
    os.makedirs(download_folder)

##### Download
################################################################################

# Ignore "Missing CF-netCDF variable" warnings from download
warnings.simplefilter("ignore", UserWarning)

tic = time.time()

source_id_dict = download_dict[source_id]
variable_dict = source_id_dict["variable_dict"]

query = {
    "source_id": source_id,
    "member_id": member_id,
    "frequency": source_id_dict["frequency"],
}

for variable_id, variable_id_dict in source_id_dict["variable_dict"].items():
    # if dictionary specifies not to include this variable, skip
    if variable_id_dict["include"] is False:
        continue

    ### SET UP QUERY
    query["variable_id"] = variable_id
    query["table_id"] = variable_id_dict["table_id"]

    if (
        "ocean_variable" in variable_id_dict.keys()
    ):  # this originally included "or EC-Earth3" (lower-res version)
        query["grid_label"] = "gr"
    else:
        query["grid_label"] = "gn"

    print("\n\n{}: ".format(variable_id), end="", flush=True)

    ### SET UP FILE PATHS
    # Paths for each plevel (None indicates surface variable)
    fpaths_tripolar = {}
    fpaths_latlon = {}
    fpaths = {}

    # TODO: automate this casting to list check
    if variable_id_dict["plevels"] is None:
        variable_id_dict["plevels"] = [None]

    skip = {}  # Whether to skip each plevel variable?
    fpaths_latlon_existing = []

    for plevel in variable_id_dict["plevels"]:
        fname = variable_id
        if plevel is not None:  # if not surface variable
            if plevel == -1:  # if seafloor variable
                fname += "_seafloor"
            else:  # else other pressure level
                # suffix for the pressure level in hPa
                fname += "{:.0f}".format(plevel / 100)

        if do_crop:
            fname = fname + "_" + tuples_to_string(lats, lons) + ".nc"
        else:
            fname = fname + ".nc"
        fpaths[plevel] = os.path.join(download_folder, fname)

        # Intermediate lat-lon file before CDO regridding
        fpaths_tripolar[plevel] = os.path.join(download_folder, fname + "_tripolar.nc")

        if do_regrid:
            fname += "_latlon"
            # TODO: check that this is actually writing anywhere
            fpaths_latlon[plevel] = os.path.join(download_folder, fname + ".nc")

            if os.path.exists(fpaths_latlon[plevel]):
                if do_overwrite:
                    print("removing existing file... ", end="", flush=True)
                    os.remove(fpaths_latlon[plevel])
                    skip[plevel] = False
                else:
                    print("skipping existing file... ", end="", flush=True)
                    skip[plevel] = True
                    fpaths_latlon_existing.append(fpaths_latlon[plevel])
            else:
                skip[plevel] = False

    # TODO: ??
    skipall = all([skip_bool for skip_bool in skip.values()])
    if skipall:
        print(
            "skipping due to existing files {}".format(fpaths_latlon_existing),
            end="",
            flush=True,
        )
        continue

    ### DOWNLOAD
    if do_download:
        print("searching ESGF... ", end="", flush=True)
        results = []

        for experiment_id in source_id_dict["experiment_ids"]:
            query["experiment_id"] = experiment_id

            experiment_id_results = []
            for data_node in source_id_dict["data_nodes"]:
                query["data_node"] = data_node
                experiment_id_results.extend(esgf_search(**query))

                # Keep looping over possible data nodes until the experiment data is found
                if len(experiment_id_results) > 0:
                    print("found {}, ".format(experiment_id), end="", flush=True)
                    results.extend(experiment_id_results)
                    break  # Break out of the loop over data nodes

        results = list(set(results))
        print("found {} files. ".format(len(results)), end="", flush=True)

        for plevel in variable_id_dict["plevels"]:
            if plevel is not None:
                if plevel == -1:
                    print("extracting seafloor values, ", end="", flush=True)
                else:
                    print(
                        "extracting {} hPa, ".format(plevel / 100), end="", flush=True
                    )

            if skip[plevel]:
                print(
                    "skipping this plevel due to existing file {}".format(
                        fpaths_latlon[plevel]
                    ),
                    end="",
                    flush=True,
                )
                continue

            print("loading metadata... ", end="", flush=True)

            if do_download_ind:  # download individual files (by time)
                seafloor_indices = None

                print("downloading individual files... ", end="", flush=True)
                ind_download_folder = os.path.join(download_folder, variable_id)
                if not os.path.exists(ind_download_folder):
                    os.makedirs(ind_download_folder)

                for i, result in enumerate(results):
                    date = result.split("_")[-1].split(".")[0]  # name?
                    ind_fname = (
                        fname.split(".")[0] + "_" + date + ".nc"
                    )  # name of individual file
                    save_fp = os.path.join(ind_download_folder, ind_fname)

                    # check if file already exists before continuing
                    # TODO: check that this continue breaks the loop as required. GPT says yes
                    if os.path.exists(save_fp):
                        print(
                            f"\n{i}: skipping {ind_fname} due to existing file: {save_fp}",
                            end="",
                            flush=True,
                        )
                        continue

                    ds = xa.open_dataset(
                        result,
                        decode_times=True,
                        # chunks={'time': '499MB'}
                        chunks={"i": 200, "j": 200},
                        # chunks = {"lev": "499MB"}
                        # chunks=None
                    ).isel(
                        lev=slice(0, 2)
                    )  # PROBLEM: too shallow

                    if plevel is not None:
                        if plevel == -1:  # if seafloor level
                            # if seafloor indices for region not yet calculated, calculate
                            if seafloor_indices is None:
                                print(
                                    "determining seafloor indices... ",
                                    end="",
                                    flush=True,
                                )
                                seafloor_indices = gen_seafloor_indices(
                                    ds.isel(time=0), var=variable_id
                                )
                                seafloor_indices = np.broadcast_to(
                                    seafloor_indices,
                                    (len(ds.time), len(ds.j), len(ds.i)),
                                )

                            # index to get seafloor values
                            print("\textracting seafloor values...", end="", flush=True)
                            cmip6_seafloor_array = extract_seafloor_vals(
                                ds[variable_id], seafloor_indices
                            )
                            ds = ds.copy()[["time", "i", "j"]]
                            ds[variable_id] = (["time", "j", "i"], cmip6_seafloor_array)

                        else:
                            cmip6_da = cmip6_da.sel(
                                plev=plevel
                            )  # this would have to be changed for models with different vertical coordinate names. Possibly in preprocessing

                    # TODO: overwrite saved file with selected level
                    if do_crop:
                        ds = _spatial_crop(ds)

                    print(
                        f"\n{i}: saving {ind_fname} to .nc file: {save_fp}",
                        end="",
                        flush=True,
                    )
                    ds.to_netcdf(save_fp)
            # major issues here with handling over-large files. Commented is what I've tried so far
            else:
                print("Problems with handling large files")
                # test_xa = print(xa.open_dataset(results[0], decode_times=False))
                # Avoid 500MB DAP request limit
                # cmip6_da = xa.open_mfdataset(
                #     results[:2],
                #     # combine="by_coords",
                #     combine="nested",
                #     concat_dim="time",
                #     # chunks={"time": "499MB"},
                #     decode_times=False,
                #     # decode_times=True,
                #     preprocess=_spatial_crop,
                #     chunks={"time": "499MB"},
                # )[variable_id]
                static_da = (
                    xa.open_dataset(results[0])[variable_id].isel(time=0).compute()
                )
                cmip6_da = xa.open_mfdataset(
                    results[:3],
                    combine="by_coords",
                    # chunks={'time': '499MB'}
                    chunks=None,
                )[variable_id]

                seafloor_indices = None

                if plevel is not None:
                    if plevel == -1:
                        if seafloor_indices is None:
                            seafloor_indices = gen_seafloor_indices(static_da)

                        cmip6_da = extract_seafloor_vals(cmip6_da, seafloor_indices)

                    else:
                        cmip6_da = cmip6_da.sel(
                            plev=plevel
                        )  # this would have to be changed for models with different vertical coordinate names. Possibly in preprocessing

                print("downloading with xarray... ", end="", flush=True)
                cmip6_da.compute()

                # TODO: change dtype of variable to float32
                print("saving to regrid via cdo... ", end="", flush=True)
                cmip6_da.to_netcdf(fpaths_tripolar[plevel])

    if do_regrid:
        # initialise instance of CDO for regridding
        cdo = Cdo()
        for plevel in variable_id_dict["plevels"]:
            if skip[plevel]:
                print(
                    "skipping this plevel due to existing file {}".format(
                        fpaths_latlon[plevel]
                    ),
                    end="",
                    flush=True,
                )
                continue

            regrid_fname = fpaths_latlon[plevel] + ".nc"  # check this
            # TODO: allow customisation of regridding method and resolution
            # currently using bilinear interpolation to 0.25x0.25 degree (native) grid
            cdo.remapbil(
                "r1440x720", input=fpaths_tripolar[plevel], output="regrid_fname"
            )
            # TODO: check if this actually removes files
            if delete_tripolar_data:
                os.remove(fpaths_tripolar[plevel])

    if do_concat_by_time:
        # TODO: get rid of useless variable_id loop: covered by parent loop
        conc_var_dir = os.path.join(download_folder, "concatted_vars")
        if not os.path.exists(conc_var_dir):
            os.makedirs(conc_var_dir)

        # for variable_id in list(source_id_dict["variable_dict"].keys()):
        #     print(variable_id)
        # print("source_id_dict", source_id_dict["variable_dict"].keys())
        # get variable subdirs

        if do_crop:
            conc_fname = variable_id + "_" + tuples_to_string(lats, lons) + ".nc"
        else:
            conc_fname = variable_id + ".nc"
        print("conc_fname", conc_fname)
        save_fp = os.path.join(conc_var_dir, conc_fname)
        print("save_path", save_fp)

        # if dir containing individual files not found
        if not os.path.exists(variable_dir):
            print(f"{variable_dir} does not exist, skipping", end="", flush=True)
            continue
        # if concatenated file already exists, don't do anything
        elif os.path.exists(save_fp):
            print(
                f"concatenated file already exists at {save_fp}",
                end="",
                flush=True,
            )
        # otherwise, concatenate and save to file
        else:
            print("concatenating files by time... ", end="", flush=True)
            nc_fs = list(Path(ind_download_folder).glob("*.nc"))
            concatted = xa.open_mfdataset(nc_fs).convert_calendar(
                "gregorian", dim="time"
            )  # may not be universal for all models

            print(f"saving concatenated file to {save_fp}... ", end="", flush=True)
            concatted.to_netcdf(save_fp)

        # for variable_id in list(source_id_dict["variable_dict"].keys()):
        #     print("variable_id", variable_id)
        #     # directory with individual files for single variable
        #     variable_dir = os.path.join(download_folder, variable_id)   # ind_download_folder
        #     print("variable_dir", variable_dir)

        #     if do_crop:
        #         fname = variable_id + "_" + tuples_to_string(lats, lons) + ".nc"
        #     else:
        #         fname = variable_id + ".nc"
        #     print("fname", fname)
        #     save_fp = os.path.join(conc_var_dir, fname)
        #     print("save_path", save_fp)

        #     # if dir containing individual files not found
        #     if not os.path.exists(variable_dir):
        #         print(f"{variable_dir} does not exist, skipping", end="", flush=True)
        #         continue
        #     # found files to concatenate
        #     else:
        #         # if concatenated file already exists, don't do anything
        #         if os.path.exists(save_fp):
        #             print(
        #                 f"concatenated file already exists at {save_fp}",
        #                 end="",
        #                 flush=True,
        #             )
        #         # concatenate and save file
        #         else:
        #             print("concatenating files by time... ", end="", flush=True)
        #             nc_fs = list(Path(variable_dir).glob("*.nc"))
        #             concatted = xa.open_mfdataset(nc_fs)

        #             print(variable_dir)
        #             print(save_fp)
        #             print(concatted)
        #             # decode time
        #             concatted = concatted.convert_calendar(
        #                 "gregorian", dim="time"
        #             )  # may not be universal for all models
        #             print(
        #                 f"saving concatenated file to {save_fp}... ", end="", flush=True
        #             )
        #             concatted.to_netcdf(save_fp)

        #     if delete_latlon_data:
        #         os.remove(fpaths_latlon[plevel])

if do_merge_by_vars:
    concatted_var_dir = os.path.join(download_folder, "concatted_vars")
    # TODO: remove assertion
    assert os.path.exists(concatted_var_dir) and os.path.isdir(
        concatted_var_dir
    ), f"{concatted_var_dir} does not exist or is not a directory"
    var_nc_fps = list(Path(concatted_var_dir).glob("*.nc"))

    vars = [str(var_nc_fp.name).split("_")[0] for var_nc_fp in var_nc_fps]
    merged_name = "_".join(vars) + ".nc"
    merged_fp = os.path.join(download_folder, merged_name)

    if not os.path.exists(merged_fp):
        print(f"merging variable files... ", end="", flush=True)
        dss = [xa.open_dataset(fname) for fname in var_nc_fps]
        merged = xa.merge(dss)
        merged.to_netcdf(merged_fp)
    else:
        print(
            f"merged file already exists at {concatted_fp}",
            end="",
            flush=True,
        )

print("\nDone.\n\n")


dur = time.time() - tic
print("\n\nTOTAL DURATION: {:.0f}m:{:.0f}s\n".format(np.floor(dur / 60), dur % 60))

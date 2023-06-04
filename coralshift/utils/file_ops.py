from __future__ import annotations

from pathlib import Path
from tqdm import tqdm
import urllib
import json
import xarray as xa
import pandas as pd
import geopandas as gpd

from coralshift.processing import spatial_data


def guarantee_existence(path: Path | str) -> Path:
    """Checks if string is an existing directory path, else creates it

    Parameter
    ---------
    path (str)

    Returns
    -------
    Path
        pathlib.Path object of path
    """
    path_obj = Path(path)
    if not path_obj.exists():
        path_obj.mkdir(parents=True)
    return path_obj.resolve()


def check_file_exists(
    filepath: Path | str = None,
    dir_path: Path | str = None,
    filename: str = None,
    suffix: str = None,
) -> bool:
    """Check if a file with the specified filename and optional suffix exists in the given directory.
    # TODO: potentially more checking required
    Parameters
    ----------
    dir_path (Path | str): Path to the directory where the file should be located.
    filename (str): Name of the file to check for.
    suffix (str, optional) Optional suffix for the filename, by default None.

    Returns
    -------
    bool: True if the file exists, False otherwise.
    """
    # if filepath argument not provided, try to create from directory path and filename
    if not filepath:
        filepath = Path(dir_path) / filename
    # if suffix argument provided (likely in conjunction with a "filename", append)
    if suffix:
        filepath = Path(filepath).with_suffix(pad_suffix(suffix))
    return filepath.is_file()


def prune_file_list_on_existence(file_list: list[Path | str]) -> list:
    """Given a list of file paths, remove any paths that do not exist on disk.

    Parameters
    ----------
        file_list (list[Path | str]): A list of file paths.

    Returns
    -------
        list[Path | str]: A list of file paths that exist on disk.
    """
    # filter out files that do not exist on disk
    existing_files = [p for p in file_list if Path(p).exists()]

    # identify and remove non-existent files from the original list
    removed_files = set(file_list) - set(existing_files)
    for removed_file in removed_files:
        file_list.remove(removed_file)
        print(f"{removed_file} not found. Removed it from list.")

    return existing_files


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path, loading_bar: bool = True) -> None:
    print("\n")
    with DownloadProgressBar(
        unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]
    ) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
    print(f"Download to {output_path} complete.")


def check_exists_download_url(
    filepath: Path | str, url: str, loading_bar: bool = True
) -> None:
    """Download a file from a URL to a given filepath, with the option to display a loading bar.

    Parameters
    ----------
        filepath (Path | str): future path at which the downloaded file will be stored
        url (str): URL from which to download the file
        loading_bar (bool, optional): Whether to display a loading bar to show download progress. Defaults to True.

    Returns
    -------
        None
    """
    # if not downloaded
    if not Path(filepath).is_file():
        # download with loading bar
        if loading_bar:
            download_url(url, str(filepath))
        # choose to download without loading bar
        else:
            urllib.request.urlretrieve(url, filename=filepath)
    # if already downloaded
    else:
        print(f"Already exists: {filepath}")


def get_n_last_subparts_path(path: Path | str, n: int) -> Path:
    """Returns 'n' last parts of a path. E.g. /first/second/third/fourth with n = 3 will return second/third/fourth"""
    return Path(*Path(path).parts[-n:])


def check_path_suffix(path: Path | str, comparison: str) -> bool:
    """Checks whether path provided ends in a particular suffix e.g. "nc". Since users usually forget to specify ".",
    pads "comparison" with a period if missing.

    Parameters
    ----------
    path (Path | str): path to have suffix checked
    comparison (str): extension to check for

    Returns
    -------
    bool: True if file path extension is equal to comparison, False otherwise"""
    p = Path(path)

    # pad with leading "."
    if "." not in comparison:
        comparison = "." + comparison

    if p.suffix == comparison:
        return True
    else:
        return False


def load_merge_nc_files(
    nc_dir: Path | str, incl_subdirs: bool = True, concat_dim: str = "time"
):
    """Load and merge all netCDF files in a directory.

    Parameters
    ----------
        nc_dir (Path | str): directory containing the netCDF files to be merged.

    Returns
    -------
        xr.Dataset: merged xarray Dataset object containing the data from all netCDF files.
    """
    # specify whether searching subdirectories as well
    files = return_list_filepaths(nc_dir, ".nc", incl_subdirs)
    # if only a single file present (no need to merge)
    if len(files) == 1:
        return xa.open_dataset(files[0])
    # combine nc files by time
    ds = xa.open_mfdataset(
        files,
        decode_cf=False,
        concat_dim=concat_dim,
        combine="nested",
        coords="minimal",
    )
    return xa.decode_cf(ds).sortby("time", ascending=True)


def merge_nc_files_in_dirs(parent_dir: Path | str, concat_dim: str = "time"):
    """Load and merge all netCDF files in a directory.

    Parameters
    ----------
        nc_dir (Path | str): directory containing the netCDF files to be merged.

    Returns
    -------
        xr.Dataset: merged xarray Dataset object containing the data from all netCDF files.
    """
    dirs = [d for d in Path(parent_dir).iterdir() if d.is_dir()]
    for dir in tqdm(dirs):
        merged_name = f"{str(dir.stem)}_time_merged.nc"
        merged_path = dir / merged_name
        print(f"Merging .nc files into {merged_path}")

        # if merged doesn't already exist
        if not merged_path.is_file():
            files = return_list_filepaths(dir, ".nc", incl_subdirs=False)
            if len(files) == 1:
                ds = xa.open_dataset(files[0])
            else:
                # combine nc files by time
                ds = xa.open_mfdataset(
                    files, decode_cf=False, concat_dim=concat_dim, combine="nested"
                ).sortby("time", ascending=True)
            ds.to_netcdf(merged_path)
        else:
            print(f"{merged_path} already exists.")


def merge_from_dirs(parent_dir: Path | str, concat_files_common: str):
    """Merging iteratively to speed up"""
    files_to_merge = list(Path(parent_dir).glob("**/*" + concat_files_common))

    merged_data = None
    for path in files_to_merge:
        dataset = xa.open_dataset(path)
        if merged_data is None:
            merged_data = dataset
        else:
            merged_data = xa.merge([merged_data, dataset])

    return merged_data

    # # specify whether searching subdirectories as well
    # files = return_list_filepaths(nc_dir, ".nc", incl_subdirs)
    # # if only a single file present (no need to merge)
    # if len(files) == 1:
    #     return xa.open_dataset(files[0])
    # # combine nc files by time
    # ds = xa.open_mfdataset(
    #     files, decode_cf=False, concat_dim=concat_dim, combine="nested"
    # )
    # return xa.decode_cf(ds).sortby("time", ascending=True)

    # for each subdir in turn, get list of files
    # merge files in list with distinct name
    # concatenate merged files


def merge_save_nc_files(download_dir: Path | str, filename: str):
    # read relevant .nc files and merge to return master xarray
    xa_ds = load_merge_nc_files(Path(download_dir))
    save_path = Path(Path(download_dir), filename).with_suffix(".nc")
    xa_ds.to_netcdf(save_path)
    print(f"Combined nc file written to {save_path}.")
    return xa_ds


# hopefully isn't necessary, since loading all datasets into memory is very intensive and not scalable
# def naive_nc_merge(dir: Path | str):
#     file_paths = return_list_filepaths(dir, ".nc")
#     # get names of files
#     filenames = [file_path.stem for file_path in file_paths]
#     # load in files to memory as xarray
#     das = [xa.load_dataset(file_path) for file_path in file_paths]

#     combined = xa.merge([da for da in das], compat="override")

#     # save combined file
#     save_name = filenames[0] + "&" + filenames[-1] + "_merged.nc"
#     save_path = Path(dir) / save_name
#     combined.to_netcdf(path=save_path)
#     print(f"{save_name} saved successfully")


def pad_suffix(suffix: str) -> str:
    """Pads the given file suffix with a leading period if necessary.

    Parameters
    ----------
        suffix (str): file suffix to pad.

    Returns
    -------
        str: The padded file suffix.
    """
    if "." not in suffix:
        suffix = "." + suffix
    return suffix


def remove_suffix(filename: str) -> str:
    """Remove the suffix from a filename or path"""
    split_list = filename.split(".")

    if len(split_list) > 2:
        raise ValueError(
            f"{filename} appears to not be a valid filname (more than one '.')"
        )
    if len(split_list) == 1:
        return filename

    return split_list[:-1][0]


def return_list_filepaths(
    files_dir: Path | str, suffix: str, incl_subdirs: bool = True
) -> list[Path]:
    """Return a list of file paths in the specified directory that have the given suffix.

    Parameters
    ----------
    files_dir (Path | str): Directory in which to search for files.
    suffix (str): File suffix to look for.
    incl_subdirs (bool, optional) If True, search for files in subdirectories as well. Defaults to False.

    Returns
    -------
    list[Path]
        List of file paths in the directory with the specified suffix.
    """
    # if searching in only specified directory, files_dir:
    if incl_subdirs:
        return list(Path(files_dir).glob("**/*" + pad_suffix(suffix)))
    # if also searching in subdirectories
    else:
        return list(Path(files_dir).glob("*" + pad_suffix(suffix)))


def read_nc_path(nc_file_path: Path | str, engine: str = "h5netcdf") -> xa.DataArray:
    """Reads single netcdf filepath and opens dataset.

    Parameters
    ----------
    nc_file_path (Path | str): path to nc file
    engine (str): specify engine with which to open file. Defaults to "h5netcdf". This isn't compatibile with all files.
        Try engine="netcdf4" if having trouble reading.
    Returns
    ---
    xa.DataArray
    """
    return xa.open_dataset(nc_file_path, engine=engine)


def dict_of_ncs_from_dir(
    dir_path: Path | str, crs: float = "epsg:4326", engine: str = "h5netcdf"
) -> dict:
    """Reads multiple netcdf files in a directory and returns a dictionary of DataArrays.

    Parameters
    ----------
    dir_path (Path | str): Path to directory containing netCDF files.
    engine (str, optional): Engine to use to read netCDF files. Defaults to "h5netcdf".

    Returns
    -------
    dict
        Dictionary containing the DataArrays, keyed by the file names without the .nc extension.

    TODO: Could also make into a more generic function which reads in different file types in correct way. Haven't done
    this for now since will all be read in different ways
    """
    # generate list of all ".nc" format files in directory
    nc_files_list = return_list_filepaths(dir_path, ".nc")

    nc_arrays_dict = {}
    for nc_path in tqdm(nc_files_list):
        # fetch "name.extension" of file from path
        path_end = get_n_last_subparts_path(nc_path, 1)
        # fetch name of file
        nc_filename = remove_suffix(str(path_end))
        # read file and assign crs
        nc_array = read_nc_path(nc_path, engine).rio.write_crs(crs)
        nc_arrays_dict[nc_filename] = nc_array

    return nc_arrays_dict


def load_gpkg(filepath):
    """Load a large gpkg file into a memory-efficient geopandas.GeoDataFrame object.

    Parameters
    ----------
        filepath (str): The filepath of the gpkg file.

    Returns
    -------
        geopandas.GeoDataFrame: A GeoDataFrame object containing the data from the gpkg file.
    """
    gdf = gpd.read_file(filepath, driver="GPKG")
    return gdf


def check_pkl_else_read_gpkg(files_dir: Path | str, filename: str) -> pd.DataFrame:
    """Load pickled pandas DataFrame if available, else load a geopackage file as a DataFrame. If neither file found,
    raisea FileNotFoundError

    Parameters
    ----------
    files_dir (Path | str): path to directory containing files
    filename (str): name of path

    Returns
    -------
    pd.DataFrame: and completion message detailing which file format was read
    """
    filename = remove_suffix(filename)
    pkl_path = Path(Path(files_dir), filename).with_suffix(".pkl")
    if pkl_path.is_file():
        print(f"Reading {pkl_path}")
        df_out = pd.read_pickle(pkl_path)
        print(f"{filename} read from {filename}.pkl")
        return df_out

    gpkg_path = Path(Path(files_dir), filename).with_suffix(".gpkg")
    if gpkg_path.is_file():
        print(f"Reading {gpkg_path}")
        df_out = load_gpkg(gpkg_path)
        # write to pkl for faster access next time
        df_out.to_pickle(pkl_path)
        print(
            f"{filename} read from {filename}.gpkg. {filename}.pkl created in same directory."
        )
        return df_out

    raise FileNotFoundError(f"{filename}.pkl/gpkg not found in {files_dir}")


def read_write_nc_file(
    files_dir: Path | str,
    filename: str,
    raster_array: xa.Dataset = None,
    ymin: float = None,
    ymax: float = None,
    xmin: float = None,
    xmax: float = None,
    resolution: float = None,
    name: str = None,
) -> xa.Dataset:
    """Read or write a NetCDF file with xarray, based on whether it already exists in the specified directory.
    If the file exists, read it and return its contents as an xarray dataset.
    If it doesn't exist, create it by converting a provided raster array to an xarray dataset, and save it to the
    specified directory.

    Parameters
    ----------
    files_dir (str | Path): The directory where the file should be read from or written to.
    filename (str): The name of the file, without the ".nc" extension.
    raster_array (xarray.Dataset, optional) A raster dataset to be converted to an xarray dataset and saved as the new
        NetCDF file. If not provided, the function assumes that the file already exists and simply reads it.
    y_bounds (tuple(float)): The upper and lower bounds of the y-axis range of the dataset.
    x_bounds (tuple(float)): The leftmost and rightmost bounds of the x-axis range of the dataset.
    resolution (float): The resolution of the dataset in units of meters.
    name (str): The name of the dataset.

    Returns
    -------
    xarray.Dataset: The contents of the NetCDF file as an xarray dataset.
    TODO: this function probably does too-separate jobs
    """
    filepath = Path(Path(files_dir), filename).with_suffix(".nc")
    # if file exists, read
    if filepath.is_file():
        return xa.open_dataset(filepath)

    # if file doesn't exist, make it from raster input
    if not filepath.is_file():
        print(f"{filename} not found in {files_dir}.")
        # if raster_array provided
        if raster_array:
            rasterized_ds = spatial_data.xa_array_from_raster(
                raster_array, (ymin, ymax), (xmin, xmax), resolution, name
            )
            rasterized_ds.to_netcdf(filepath)
            return rasterized_ds


def add_suffix_if_necessary(filepath: Path | str, suffix_to_add: str) -> Path:
    """Add a suffix to a file path if it doesn't already have it.

    Parameters
    ----------
    filepath (Path | str): the file path to check.
    suffix_to_add (str): the suffix to add if necessary.

    Returns
    -------
    Path: the modified file path with the correct suffix.

    Raises
    ------
    ValueError: if the file path already has a suffix that conflicts with the suffix_to_add.
    """
    filepath = Path(filepath)
    # if correct suffix, leave as is
    if filepath.suffix == pad_suffix(suffix_to_add):
        return Path(filepath)
    # if no suffix
    if filepath.suffix == "":
        return filepath.with_suffix(pad_suffix(suffix_to_add))
    else:
        raise ValueError(
            f"{filepath} terminates in a conflicting suffix to suffix_to_add: {suffix_to_add}."
        )


def generate_filepath(
    dir_path: str | Path, filename: str = None, suffix: str = None
) -> Path:
    """Generates directory path if non-existant; if filename provided, generates filepath, adding suffix if
    necessary."""
    # if generating/ensuring directory path
    if not filename:
        return guarantee_existence(dir_path)
    # if filename provided, seemingly with suffix included
    elif not suffix:
        return Path(dir_path) / filename
    # if filename and suffix provided
    else:
        return (Path(dir_path) / filename).with_suffix(suffix)


def save_json(
    json_file: dict, filepath: str | Path, indent: int = 4, verbose: bool = True
) -> None:
    """Save a JSON object to a file.

    Parameters
    ----------
    json_file (dict): JSON object to be saved.
    filepath (str | Path): Path to the output file.
    indent (int, optional): Indentation level for the JSON file (default is 4).

    Returns
    -------
    None
    """
    # Serializing json
    json_object = json.dumps(json_file, indent=indent)

    with open(str(filepath), "w") as outfile:
        outfile.write(json_object)

    if verbose:
        print(f"Dictionary saved as json file at {filepath}")

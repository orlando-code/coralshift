from __future__ import annotations

from pathlib import Path
from tqdm import tqdm
import urllib
import json
import xarray as xa
import pandas as pd
import geopandas as gpd
import rasterio
import rioxarray as rio
import numpy as np

from coralshift.processing import spatial_data
from coralshift.utils import utils, directories


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
    filename = remove_suffix(utils.replace_dot_with_dash(filename))
    save_path = (Path(save_dir) / filename).with_suffix(".nc")
    if not save_path.is_file():
        print(f"Writing {filename} to file at {save_path}")
        spatial_data.process_xa_d(xa_d).to_netcdf(save_path)
        print("Writing complete.")
    else:
        print(f"{filename} already exists in {save_dir}")

    if return_array:
        return save_path, xa.open_dataset(save_path)
    else:
        return save_path


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


def xa_d_from_geojson(geojson_path: Path | str) -> xa.DataArray:
    """
    Load a GeoJSON file and convert it to an xarray DataArray.

    Parameters
    ----------
        geojson_path (Path or str): Path to the GeoJSON file.

    Returns
    -------
        xa.DataArray: xarray DataArray representing the GeoJSON data.
    """
    # read geojson
    gdf = gpd.read_file(geojson_path)
    # Convert the GeoDataFrame to a DataFrame
    df = gdf.drop("geometry", axis=1)
    # Convert the DataFrame to an xarray Dataset TODO: chunking
    return xa.Dataset.from_dataframe(df)


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
        print(f"{filepath.stem} already exists in {filepath.parent}")


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


# def load_merge_nc_files(
#     nc_dir: Path | str, incl_subdirs: bool = True, concat_dim: str = "time"
# ):
#     """Load and merge all netCDF files in a directory.

#     Parameters
#     ----------
#         nc_dir (Path | str): directory containing the netCDF files to be merged.

#     Returns
#     -------
#         xr.Dataset: merged xarray Dataset object containing the data from all netCDF files.
#     """
#     # specify whether searching subdirectories as well
#     files = return_list_filepaths(nc_dir, ".nc", incl_subdirs)
#     # if only a single file present (no need to merge)
#     if len(files) == 1:
#         return xa.open_dataset(files[0])
#     # combine nc files by time
#     ds = xa.open_mfdataset(
#         files,
#         decode_cf=False,
#         concat_dim=concat_dim,
#         combine="nested",
#         coords="minimal",
#     )
#     return xa.decode_cf(ds).sortby("time", ascending=True)


def merge_nc_files_in_dir(
    nc_dir: Path | str,
    filename: str = None,
    merged_save_path: Path | str = None,
    incl_subdirs: bool = False,
    concat_dim: str = "time",
    format: str = ".nc",
):
    """
    Load and merge all netCDF files in a directory.

    Parameters
    ----------
        nc_dir (Path or str): The directory containing the netCDF files to be merged.
        filename (str, optional): The desired filename for the merged netCDF file. If not specified, a default filename
            is used.
        merged_save_path (Path or str, optional): The path to save the merged netCDF file. If not specified, it will be
            saved in the same directory as `nc_dir`.
        incl_subdirs (bool, optional): Specifies whether to search for netCDF files in subdirectories as well. Default
            is False.
        concat_dim (str, optional): The name of the dimension along which the netCDF files will be concatenated.
            Default is "time".
        format (str, optional): The format of the netCDF file. Default is ".nc".

    Returns
    -------
        merged_save_path (Path): The path where the merged netCDF file is saved.
    """
    # specify whether searching subdirectories as well
    filepaths = return_list_filepaths(nc_dir, format, incl_subdirs)
    # if only a single file present (no need to merge)
    if len(filepaths) == 1:
        print(f"Single {format} file found in {str(nc_dir)}.")
        return xa.open_dataset(filepaths[0])

    nc_dir = Path(nc_dir)

    # if filename not specified
    if not filename:
        filename = f"{nc_dir.parent.stem}_merged"

    # if path to save file not specified
    if not merged_save_path:
        merged_save_path = Path(nc_dir / remove_suffix(filename)).with_suffix(".nc")

    # check if already merged
    if not merged_save_path.is_file():
        print(f"Merging {format} files into {merged_save_path}")
        merged_ds = spatial_data.process_xa_d(
            xa.open_mfdataset(filepaths, concat_dim=concat_dim, combine="nested")
        )
        merged_ds.to_netcdf(merged_save_path)
    else:
        print(f"{merged_save_path} already exists.")
    return merged_save_path

    # # combine nc files by time
    # ds = xa.open_mfdataset(
    #     filepaths,
    #     decode_cf=False,
    #     concat_dim=concat_dim,
    #     combine="nested",
    #     coords="minimal",
    # )
    # return xa.decode_cf(ds).sortby("time", ascending=True)


def merge_nc_files_in_dirs(
    parent_dir: Path | str, filename: str = "placeholder", concat_dim: str = "time"
):
    """
    Load and merge all netCDF files in multiple directories.
    # TODO: test
    Parameters
    ----------
        parent_dir (Path or str): The parent directory containing the directories with netCDF files to be merged.
        concat_dim (str, optional): The name of the dimension along which the netCDF files will be concatenated.
            Default is "time".

    Returns
    -------
        xarray.Dataset: The merged xarray Dataset object containing the data from all netCDF files.
    """
    nc_dirs = [d for d in Path(parent_dir).iterdir() if d.is_dir()]
    for nc_dir in tqdm(nc_dirs):
        return merge_nc_files_in_dir(
            nc_dir, filename, include_subdirs=True, concat_dim=concat_dim
        )
        # merged_name = f"{str(dir.stem)}_time_merged.nc"
        # merged_path = dir / merged_name
        # print(f"Merging .nc files into {merged_path}")

        # # if merged doesn't already exist
        # if not merged_path.is_file():
        #     files = return_list_filepaths(dir, ".nc", incl_subdirs=False)
        #     if len(files) == 1:
        #         ds = xa.open_dataset(files[0])
        #     else:
        #         # combine nc files by time
        #         ds = xa.open_mfdataset(
        #             files, decode_cf=False, concat_dim=concat_dim, combine="nested"
        #         ).sortby("time", ascending=True)
        #     ds.to_netcdf(merged_path)
        # else:
        #     print(f"{merged_path} already exists.")


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


# def merge_save_nc_files(download_dir: Path | str, filename: str):
#     # read relevant .nc files and merge to return master xarray
#     xa_ds = load_merge_nc_files(Path(download_dir))
#     save_path = Path(Path(download_dir), filename).with_suffix(".nc")
#     xa_ds.to_netcdf(save_path)
#     print(f"Combined nc file written to {save_path}.")
#     return xa_ds


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
    dir_path: Path | str, crs: str = "epsg:4326", engine: str = "h5netcdf"
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
        nc_array = read_nc_path(nc_path, engine).rio.write_crs(crs, inplace=True)
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
    return gpd.read_file(filepath, driver="GPKG")


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
    pkl_path = (Path(files_dir) / filename).with_suffix(".pkl")
    if pkl_path.is_file():
        print(f"Reading {pkl_path}")
        df_out = pd.read_pickle(pkl_path)
        print(f"{filename} data read from {filename}.pkl")
        return df_out

    gpkg_path = (Path(files_dir) / filename).with_suffix(".gpkg")
    if gpkg_path.is_file():
        print(f"Reading {gpkg_path}")
        df_out = load_gpkg(gpkg_path)
        # write to pkl for faster access next time
        df_out.to_pickle(pkl_path)
        print(
            f"{filename} data read from {filename}.gpkg. {filename}.pkl created in same directory for rapid access."
        )
        return df_out

    raise FileNotFoundError(
        f"Neither {filename}.pkl nor {filename}.gpkg not found in {files_dir}"
    )


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
        return (Path(dir_path) / filename).with_suffix(pad_suffix(suffix))


class NpEncoder(json.JSONEncoder):
    """
    Custom JSON encoder to handle NumPy types.

    This class extends the `json.JSONEncoder` class and provides custom handling for NumPy types, including
    `np.integer`, `np.floating`, and `np.ndarray`. It converts these types to their corresponding Python types to
    ensure proper JSON serialization.

    Usage:
    To use this custom encoder, create an instance of `NpEncoder` and pass it as the `cls` parameter when calling
    `json.dumps()` or `json.dump()`
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


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
    json_object = json.dumps(json_file, indent=indent, cls=NpEncoder)

    with open(str(filepath), "w") as outfile:
        outfile.write(json_object)

    if verbose:
        print(f"Dictionary saved as json file at {filepath}")


def tifs_to_ncs(nc_dir: Path | list[str], target_resolution_d: float = None) -> None:
    tif_dir = nc_dir.parent
    tif_paths = return_list_filepaths(tif_dir, ".tif")
    for tif_path in tqdm(
        tif_paths, total=len(tif_paths), desc="Writing tifs to nc files"
    ):
        # filename = str(file_ops.get_n_last_subparts_path(tif, 1))
        filename = tif_path.stem
        tif_array = tif_to_xa_array(tif_path)
        # xa_array_dict[filename] = tif_array.rename(filename)
        if target_resolution_d:
            tif_array = spatial_data.upsample_xarray_to_target(
                xa_array=tif_array, target_resolution=target_resolution_d
            )
        # save array to nc file
        save_nc(tif_dir, filename, tif_array)

    return tif_paths
    print(f"All tifs converted to xarrays and stored as .nc files in {nc_dir}.")


def tif_to_xa_array(tif_path) -> xa.DataArray:
    return spatial_data.process_xa_d(rio.open_rasterio(rasterio.open(tif_path)))


def save_dict_xa_ds_to_nc(
    xa_d_dict: dict, save_dir: Path | str, target_resolution: float = None
) -> None:
    for filename, array in tqdm(xa_d_dict.items(), desc="Writing tifs to nc files"):
        if target_resolution:
            spatial_data.upsample_xarray_to_target(
                xa_array=array, target_resolution=target_resolution
            )
        save_nc(save_dir, filename, array)


def resample_list_xa_ds_to_target_res_and_save(
    xa_das: list[xa.DataArray],
    target_resolution_d: float,
    unit: str = "m",
    lat_lims: tuple[float] = (-10, -17),
    lon_lims: tuple[float] = (142, 147),
) -> None:
    """
    Resamples a list of xarray DataArrays to a target resolution, and saves the resampled DataArrays to NetCDF files.

    Parameters
    ----------
        xa_das (list[xa.DataArray]): A list of xarray DataArrays to be resampled.
        target_resolution_d (float): The target resolution in degrees or meters, depending on the unit specified.
        unit (str, optional): The unit of the target resolution. Defaults to "m".
        lat_lims (tuple[float], optional): Latitude limits for the dummy DataArray used for resampling.
            Defaults to (-10, -17).
        lon_lims (tuple[float], optional): Longitude limits for the dummy DataArray used for resampling.
            Defaults to (142, 147).

    Returns
    -------
        None
    """

    dummy_xa = spatial_data.generate_dummy_xa(target_resolution_d, lat_lims, lon_lims)

    save_dir = generate_filepath(
        (
            directories.get_comparison_dir()
            / utils.replace_dot_with_dash(f"{target_resolution_d:.05f}d_arrays")
        )
    )

    for xa_da in tqdm(
        xa_das,
        desc=f"Resampling xarray DataArrays to {target_resolution_d:.05f}d",
        position=1,
        leave=True,
    ):
        filename = utils.replace_dot_with_dash(
            f"{xa_da.name}_{target_resolution_d:.05f}d"
        )
        save_path = (save_dir / filename).with_suffix(".nc")

        if not save_path.is_file():
            xa_resampled = spatial_data.process_xa_d(
                spatial_data.upsample_xa_d_to_other(
                    spatial_data.process_xa_d(xa_da), dummy_xa, name=xa_da.name
                )
            )
            # causes problems with saving
            if "grid_mapping" in xa_resampled.attrs:
                del xa_resampled.attrs["grid_mapping"]

            xa_resampled.to_netcdf(save_path)
        else:
            print(f"{filename} already exists in {save_dir}")


def resample_list_xa_ds_to_target_res_list_and_save(
    xa_das: list[xa.DataArray],
    target_resolutions: list[float],
    units: list[str],
    lat_lims: tuple[float] = (-10, -17),
    lon_lims: tuple[float] = (142, 147),
) -> None:
    """
    Resamples a list of xarray DataArrays to multiple target resolutions specified in a list, and saves the resampled
    DataArrays to NetCDF files.

    Parameters
    ----------
        xa_das (list[xa.DataArray]): A list of xarray DataArrays to be resampled.
        target_resolutions (list[float]): A list of target resolutions in degrees or meters, depending on the units
            specified.
        units (list[str]): A list of units corresponding to the target resolutions.
        lat_lims (tuple[float], optional): Latitude limits for the dummy DataArray used for resampling.
            Defaults to (-10, -17).
        lon_lims (tuple[float], optional): Longitude limits for the dummy DataArray used for resampling. Defaults to
            (142, 147).

    Returns
    -------
        None
    """

    target_resolutions = [
        spatial_data.choose_resolution(number, string)[1]
        for number, string in zip(target_resolutions, units)
    ]
    for i, resolution in tqdm(
        enumerate(target_resolutions),
        desc="Progress through resolutions",
        position=0,
        leave=True,
        total=len(target_resolutions),
    ):
        unit = units[i]
        spatial_data.resample_list_xa_ds_to_target_res_and_save(
            xa_das, resolution, unit
        )

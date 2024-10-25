# general
import pandas as pd
import numpy as np

# spatial
import xarray as xa
import geopandas as gpd

# file ops
from pathlib import Path
from tqdm import tqdm
import urllib
import json
import yaml
import os
import pickle
import csv

# custom
from coralshift.processing import spatial_data
from coralshift.utils import utils, config


class FileName:
    def __init__(
        self,
        variable_id: str | list,
        grid_type: str,
        fname_type: str,
        date_range: str = None,
        lats: list[float, float] = None,
        lons: list[float, float] = None,
        levs: list[int, int] = None,
        plevels: list[float, float] = None,
    ):
        """
        Args:
            source_id (str): name of climate model
            member_id (str): model run
            variable_id (str): variable name
            grid_type (str): type of grid (tripolar or latlon)
            fname_type (str): type of file (individual, time-concatted, var_concatted) TODO
            lats (list[float, float], optional): latitude range. Defaults to None.
            lons (list[float, float], optional): longitude range. Defaults to None.
            plevels (list[float, float], optional): pressure level range. Defaults to None.
            fname (str, optional): filename. Defaults to None to allow construction.
        """
        self.variable_id = variable_id
        self.grid_type = grid_type
        self.fname_type = fname_type
        self.date_range = date_range
        self.lats = lats
        self.lons = lons
        self.levs = levs
        self.plevels = plevels

    def get_spatial(self):
        if self.lats and self.lons:  # if spatial range specified (i.e. cropping)
            # cast self.lats and self.lons lists to integers. A little crude, but avoids decimals in filenames
            lats = [int(lat) for lat in self.lats]
            lons = [int(lon) for lon in self.lons]
            return utils.lat_lon_string_from_tuples(lats, lons).upper()
        else:
            return "uncropped"

    def get_plevels(self):
        if self.plevels == [-1] or self.plevels == -1:  # seafloor
            return f"sfl-{max(self.levs)}"
        elif not self.plevels:
            return "sfc"
            if self.plevels[0] is None:
                return "sfc"
        elif isinstance(self.plevels, float):  # specified pressure level
            return "{:.0f}".format(self.plevels / 100)
        elif isinstance(self.plevels, list):  # pressure level range
            if self.plevels[0] is None:  # if plevels is list of None, surface
                return "sfc"
            return f"levs_{min(self.plevels)}-{max(self.plevels)}"
        else:
            raise ValueError(
                f"plevel must be one of [-1, float, list]. Instead received '{self.plevels}'"
            )

    def get_var_str(self):
        if isinstance(self.variable_id, list):
            return "_".join(self.variable_id)
        else:
            return self.variable_id

    def get_grid_type(self):
        if self.grid_type == "tripolar":
            return "tp"
        elif self.grid_type == "latlon":
            return "ll"
        else:
            raise ValueError(
                f"grid_type must be 'tripolar' or 'latlon'. Instead received '{self.grid_type}'"
            )

    def get_date_range(self):
        if not self.date_range:
            return None
        if self.fname_type == "time_concatted" or self.fname_type == "var_concatted":
            # Can't figure out how it's being done currently
            return str("-".join((str(self.date_range[0]), str(self.date_range[1]))))
        else:
            return self.date_range

    def join_as_necessary(self):
        var_str = self.get_var_str()
        spatial = self.get_spatial()
        plevels = self.get_plevels()
        grid_type = self.get_grid_type()
        date_range = self.get_date_range()

        # join these variables separated by '_', so long as the variable is not None
        return "_".join(
            [i for i in [var_str, spatial, plevels, grid_type, date_range] if i]
        )

    def construct_fname(self):

        if self.fname_type == "var_concatted":
            if not isinstance(self.variable_id, list):
                raise TypeError(
                    f"Concatted variable requires multiple variable_ids. Instead received '{self.variable_id}'"
                )

        self.fname = self.join_as_necessary()

        return f"{self.fname}.nc"


def combine_dataframes(df1, df2):
    # Get shared columns
    # shared_cols = set(df1.columns) & set(df2.columns)

    # Add new columns from df2 to df1
    new_cols_df2 = set(df2.columns) - set(df1.columns)
    for col in new_cols_df2:
        df1[col] = np.nan  # Add new column with NaN values

    # Concatenate data frames along rows
    combined_df = pd.concat([df1, df2], ignore_index=True, sort=False)

    # Reorder columns based on original header (df1)
    combined_df = combined_df.reindex(columns=df1.columns)

    return combined_df


def write_dict_to_csv(dict_info, csv_fp):
    csv_fp = Path(csv_fp)
    # read new data
    flattened_data = utils.flatten_dict(dict_info)
    df_new = pd.DataFrame([flattened_data])
    if csv_fp.exists():
        df = pd.read_csv(csv_fp)
    else:
        df_new.to_csv(csv_fp, index=False)
        return

    combined_df = combine_dataframes(df, df_new)
    combined_df.to_csv(csv_fp, index=False)


# def write_dict_to_csv(yaml_dict: dict, csv_fp: str | Path):
#     """
#     Writes data from a YAML file to a CSV file.

#     Args:
#         yaml_fp (str or Path): The file path of the YAML configuration file.

#     Raises:
#         FileNotFoundError: If the YAML file specified by `yaml_fp` does not exist.

#     """
#     flattened_data = utils.flatten_dict(yaml_dict)

#     # Create DataFrame from flattened dictionary
#     df = pd.DataFrame([flattened_data])

#     # Check if CSV file exists
#     file_exists = Path(csv_fp).exists()

#     # Read existing header from CSV file if it exists
#     existing_header = []
#     if file_exists:
#         existing_df = pd.read_csv(csv_fp)
#         existing_header = existing_df.columns.tolist()

#     # Add new columns to the DataFrame and update header
#     for col in df.columns:
#         if col not in existing_header:
#             existing_header.append(col)
#             if file_exists:
#                 existing_df[col] = pd.NA  # Add new column with NaN values

#     # Concatenate existing DataFrame and new DataFrame
#     if file_exists:
#         # df = pd.concat([existing_df, df], ignore_index=True, sort=False)
#         df = existing_df.merge(df, sort=False)

#     # Write DataFrame to CSV
#     # df.to_csv(csv_fp, mode='a', index=False, header=not file_exists)
#     df.to_csv(csv_fp, mode='a', index=False, header=existing_header)

# # Flatten nested dictionaries
# flattened_data = utils.flatten_dict(yaml_dict)

# # Extract keys and values
# keys = list(flattened_data.keys())
# values = list(flattened_data.values())

# # Check if CSV file exists
# csv_file = Path(csv_fp)
# file_exists = csv_file.exists()

# # Open CSV file in append mode
# # with open(csv_fp, "a", newline="") as csv_f:
# #     csv_writer = csv.writer(csv_f)

# #     # Write header if CSV file is newly created
# #     if not file_exists:
# #         csv_writer.writerow(keys)

# #     header = read_csv_header(csv_fp)
# #     # if keys contains values not in header, append these values to the original header (overwrite original)
# #     for key in keys:
# #         if key not in header:
# #             header.append(key)
# #     ### This line should overwrite csv header with updated header

# #     # Write values to CSV
# #     csv_writer.writerow(values)
# # Open CSV file in read mode to read the existing content
# existing_content = []
# if file_exists:
#     with open(csv_fp, "r", newline="") as csv_f:
#         reader = csv.reader(csv_f)
#         # Read existing content line by line
#         existing_content = list(reader)

# # Open CSV file in write mode to write the updated content
# with open(csv_fp, "w", newline="") as csv_f:
#     csv_writer = csv.writer(csv_f)

#     # Write the updated header
#     if not file_exists:
#         csv_writer.writerow(keys)
#     else:
#         # Update the header with new keys not present in the original header
#         header = existing_content[0]
#         for key in keys:
#             if key not in header:
#                 header.append(key)
#         csv_writer.writerow(header)

#         # Write the original data
#         for row in existing_content[1:]:
#             csv_writer.writerow(row)

#     # Write values to CSV
#     csv_writer.writerow(values)
#####

# # Create CSV file if it doesn't exist
# if not Path(csv_fp).exists():
#     with open(csv_fp, "w", newline="") as csv_f:
#         csv_writer = csv.DictWriter(csv_f, fieldnames=flattened_data.keys())
#         csv_writer.writeheader()

# # Open CSV file in append mode
# with open(csv_fp, "a", newline="") as csv_f:
#     csv_writer = csv.DictWriter(csv_f, fieldnames=flattened_data.keys())
#     csv_writer.writerow(flattened_data)
#     # make a list of values in the order of columns and write them
#     csv_writer.writerow(
#         [flattened_data.get(col, None) for col in flattened_data.keys()]
#     )


def read_csv_header(csv_fp: str | Path) -> list[str]:
    """
    Reads the header (first line) of a CSV file.

    Args:
        csv_fp (str or Path): The file path of the CSV file.

    Returns:
        list: List containing the header fields.
    """
    # Open CSV file in read mode
    with open(csv_fp, "r", newline="") as csv_f:
        csv_reader = csv.reader(csv_f)
        # Read the first row (header)
        header = next(csv_reader)

    return header


class FileHandler:
    def __init__(self, config_info, model_code, base_dir: str | Path = None):
        self.config_info = config_info
        self.model_code = model_code
        self.base_dir = base_dir

    def construct_fp_dir(self):
        res_str = utils.replace_dot_with_dash(str(self.config_info["resolution"]))
        return (
            Path(Path(self.base_dir) if self.base_dir else config.runs_dir)
            / f"{res_str}d"
            / self.model_code
        )

    def construct_fp_stem(self):
        return "_".join(self.config_info["datasets"])

    def unique_num(self, fp_dir):
        """Returns the highest ID number of the same fp_stem (function of datasets) in the IDXXX naming convention"""
        counter = 0
        fp_stem = self.construct_fp_stem()
        new_filename = f"ID000_{fp_stem}"
        while list(Path(fp_dir).glob(f"{new_filename}*")):
            counter += 1
            new_filename = f"ID{utils.pad_number_with_zeros(counter, resulting_len=3)}_{Path(fp_stem).stem}"
        return counter

    def get_highest_unique_fname(self, fp_dir=None):
        fp_stem = self.construct_fp_stem()
        if not fp_dir:
            fp_dir = self.construct_fp_dir()
        highest_num = self.unique_num(fp_dir) - 1
        return (
            f"ID{utils.pad_number_with_zeros(highest_num, resulting_len=3)}_{fp_stem}"
        )

    def get_next_unique_fname(self, fp_dir=None):
        fp_stem = self.construct_fp_stem()
        if not fp_dir:
            fp_dir = self.construct_fp_dir()
        return f"ID{utils.pad_number_with_zeros(self.unique_num(fp_dir), resulting_len=3)}_{fp_stem}"

    def get_next_unique_fp_root(self, fp_dir=None):
        if not fp_dir:
            fp_dir = self.construct_fp_dir()
        return fp_dir / self.get_next_unique_fname(fp_dir)

    def get_highest_unique_fp_root(self, fp_dir=None):
        if not fp_dir:
            fp_dir = self.construct_fp_dir()
        return fp_dir / self.get_highest_unique_fname(fp_dir)


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
        path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj.resolve()


def check_exists_save(
    xa_d: xa.Dataset | xa.DataArray,
    save_fp: str | Path,
    save_message: str,
    skip_message: str,
):
    if not os.path.exists(save_fp):
        print(save_message, flush=True)
    else:
        print(skip_message, flush=True)
    return not os.path.exists(save_fp)


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
            xa.open_mfdataset(
                filepaths,
                concat_dim=concat_dim,
                combine="nested",
                drop_variables=["depth"],
            )
        )
        merged_ds.to_netcdf(merged_save_path)
    else:
        print(f"{merged_save_path} already exists.")
    return merged_save_path


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
    files_dir: Path | str, suffix: str, incl_subdirs: bool = False
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


# def dict_of_ncs_from_dir(
#     dir_path: Path | str, crs: str = "epsg:4326", engine: str = "h5netcdf"
# ) -> dict:
#     """Reads multiple netcdf files in a directory and returns a dictionary of DataArrays.

#     Parameters
#     ----------
#     dir_path (Path | str): Path to directory containing netCDF files.
#     engine (str, optional): Engine to use to read netCDF files. Defaults to "h5netcdf".

#     Returns
#     -------
#     dict
#         Dictionary containing the DataArrays, keyed by the file names without the .nc extension.

#     TODO: Could also make into a more generic function which reads in different file types in correct way. Haven't
# done
#     this for now since will all be read in different ways
#     """
#     # generate list of all ".nc" format files in directory
#     nc_files_list = return_list_filepaths(dir_path, ".nc")

#     nc_arrays_dict = {}
#     for nc_path in tqdm(nc_files_list):
#         # fetch "name.extension" of file from path
#         path_end = get_n_last_subparts_path(nc_path, 1)
#         # fetch name of file
#         nc_filename = remove_suffix(str(path_end))
#         # read file and assign crs
#         nc_array = read_nc_path(nc_path, engine).rio.write_crs(crs, inplace=True)
#         nc_arrays_dict[nc_filename] = nc_array

#     return nc_arrays_dict


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


def resample_list_xa_ds_into_dict(
    xa_das: list[xa.DataArray],
    target_resolution: float,
    unit: str = "m",
    lat_lims: tuple[float] = (-10, -17),
    lon_lims: tuple[float] = (142, 147),
) -> dict:
    """
    Resample a list of xarray DataArrays to the target resolution and merge them.

    Parameters
    ----------
        xa_das (list[xa.DataArray]): A list of xarray DataArrays to be resampled and merged.
        target_resolution (float): The target resolution for resampling.
        unit (str, defaults to "m"): The unit of the target resolution.
        interp_method: (str, defaults to "linear") The interpolation method for resampling.

    Returns
    -------
        A dictionary containing the resampled xarray DataArrays merged by their names.
    """
    # TODO: will probably need to save to individual files/folders and combine at test/train time
    # may need to go to target array here
    target_resolution_d = spatial_data.choose_resolution(target_resolution, unit)[1]

    target_xa_d = spatial_data.generate_dummy_xa(
        target_resolution_d, lat_lims, lon_lims
    )

    resampled_xa_das_dict = {}
    for xa_da in tqdm(xa_das, desc="Resampling xarray DataArrays"):
        # xa_resampled = resample_xa_d_to_other(xa_da, dummy_xa, name=xa_da.name)
        xa_resampled = spatial_data.resample_xa_d_to_other(xa_da, target_xa_d)
        resampled_xa_das_dict[xa_da.name] = xa_resampled

    return resampled_xa_das_dict


def extract_variable(xa_d: xa.Dataset | xa.DataArray, name=None):
    """
    Extract the first data variable from an xarray dataset that isn't "spatial_ref".

    Args:
        xa_d (xa.Dataset or xa.DataArray): Input xarray dataset or data array.
        name (str, optional): Name of the specific variable to extract. Defaults to None.

    Returns:
        xa.DataArray or None: The extracted data array or None if the input object is not an xarray dataset.
    """
    if isinstance(xa_d, xa.Dataset):
        if name:
            variable = xa_d[name]
        else:
            # Find the first variable that isn't "spatial_ref"
            variables = [
                var_name
                for var_name in list(xa_d.data_vars)
                if var_name != "spatial_ref"
            ]
            if variables:
                variable = xa_d[variables[0]]
            else:
                variable = None
        return variable
    else:
        return xa_d


####
# Michaelmas


def create_coord_subdir_name(og_dir_p, lats, lons, depths):
    """
    Creates a subdirectory of the original directory with the coordinates of the data appended to the name.

    Args:
        og_dir_p (str): The original directory.
        lats (list): The latitude values of the data.
        lons (list): The longitude values of the data.
        depths (list): The depth values of the data.

    Returns:
        None

    TODO: make less specific i.e. can specify a subset of info to append to new dir. Could split into
    naming and creation
    """
    og_dir_name = Path(og_dir_p).name
    # Concatenate lat and lon values to create a subdirectory name
    sub_dir_name = utils.replace_dot_with_dash(
        f"{og_dir_name}_lat{min(lats)}_{max(lats)}_lon{min(lons)}_{max(lons)}_dep{min(depths)}_{max(depths)}"
    )

    return sub_dir_name


def create_subdirectory(og_dir_p, subdir_name):
    """
    Creates a subdirectory of the original directory with the coordinates of the data appended to the name.

    Args:
        og_dir_p (str): The original directory.
        new_dir_name (str): The name of the new subdirectory.

    Returns:
        None
    """
    # Create a Path object for the subdirectory
    subdir_p = og_dir_p / subdir_name

    # Check if the subdirectory already exists
    if not subdir_p.exists():
        # Create the subdirectory
        subdir_p.mkdir(parents=True, exist_ok=True)
        print(f"Subdirectory '{subdir_name}' created successfully.")
    else:
        print(f"Subdirectory '{subdir_name}' already exists.")


def read_yaml(yaml_path: str | Path):
    with open(yaml_path, "r") as file:
        yaml_info = yaml.safe_load(file)
    return yaml_info


def edit_yaml(yaml_path: str | Path, info: dict):
    yaml_info = read_yaml(yaml_path)
    yaml_info.update(info)

    save_yaml(yaml_path, yaml_info)


def save_yaml(yaml_path: str | Path, info: dict):
    with open(yaml_path, "w") as file:
        yaml.dump(info, file)


def uniquify_file_numerically(dir_path: str | Path, filename: str):
    """If a file already exists in the directory, make a new file with a number appended to the end"""
    counter = 1
    new_filename = f"{filename}_000"
    while (dir_path / new_filename).exists():
        new_filename = f"{Path(filename).stem}_{utils.pad_number_with_zeros(counter, resulting_len=3)}{Path(filename).suffix}"  # noqa
        counter += 1
    return new_filename


def uniquify_file_wordily(dir_path: str | Path, filename: str):
    """If a file already exists in the directory, make a new file with a letter appended to the end"""
    counter = 97  # ASCII code for 'a'
    new_filename = f"{filename}_a"
    while (dir_path / new_filename).exists():
        new_filename = f"{Path(filename).stem}_{chr(counter)}_{Path(filename).suffix}"
        counter += 1
    return new_filename


def read_pickle(pkl_path: str | Path):
    with open(pkl_path, "rb") as file:
        pkl_info = pickle.load(file)
    return pkl_info


def write_pickle(pkl_path: str | Path, info):
    with open(pkl_path, "wb") as file:
        pickle.dump(info, file)


def rename_nc_with_coords(
    nc_fp: Path | str, lat_coord_name: str = None, lon_coord_name: str = None, delete_og: bool = True
) -> None:
    """
    Renames a NetCDF file with latitude and longitude coordinates.

    Args:
        nc_fp (Path | str): The file path of the NetCDF file to be renamed.
        lat_coord_name (str, optional): The name of the latitude coordinate variable. If not provided,
            common latitude coordinate names will be used.
        lon_coord_name (str, optional): The name of the longitude coordinate variable. If not provided,
            common longitude coordinate names will be used.
        delete_og (bool, optional): Whether to delete the original file after renaming. Defaults to False.

    Raises:
        ValueError: If the latitude or longitude coordinate is not found in the dataset.
        FileExistsError: If the new file path already exists.

    Returns:
        None
    """
    nc_fp = Path(nc_fp)
    nc_xa = xa.open_dataset(nc_fp)

    lat_coord_possibilities = ["lat", "latitude", "y"] if not lat_coord_name else [lat_coord_name]
    lon_coord_possibilities = ["lon", "longitude", "x"] if not lon_coord_name else [lon_coord_name]

    lat_coord = next((coord for coord in lat_coord_possibilities if coord in nc_xa.coords), None)
    lon_coord = next((coord for coord in lon_coord_possibilities if coord in nc_xa.coords), None)

    if not lat_coord:
        raise ValueError("Latitude coordinate not found in the dataset.")
    if not lon_coord:
        raise ValueError("Longitude coordinate not found in the dataset.")

    min_lat, max_lat = nc_xa[lat_coord].values.min(), nc_xa[lat_coord].values.max()
    min_lon, max_lon = nc_xa[lon_coord].values.min(), nc_xa[lon_coord].values.max()

    lats_strs = [f"s{utils.replace_dot_with_dash(str(abs(round(lat, 1))))}" if lat < 0 else f"n{utils.replace_dot_with_dash(str(abs(round(lat, 1))))}" for lat in [min_lat, max_lat]]   # noqa
    lons_strs = [f"w{utils.replace_dot_with_dash(str(abs(round(lon, 1))))}" if lon < 0 else f"e{utils.replace_dot_with_dash(str(abs(round(lon, 1))))}" for lon in [min_lon, max_lon]]   # noqa

    new_fp = nc_fp.parent / f"{nc_fp.stem}_{lats_strs[1]}_{lats_strs[0]}_{lons_strs[0]}_{lons_strs[1]}.nc"

    if new_fp.exists():
        raise FileExistsError(f"File {new_fp} already exists.")

    print("Writing file...")
    nc_xa.to_netcdf(new_fp)
    print(f"Written {nc_fp} to {new_fp}")

    if delete_og:
        nc_fp.unlink()
        print(f"Deleted {nc_fp}")


############
# DEPRECATED
############


# def tifs_to_ncs(nc_dir: Path | list[str], target_resolution_d: float = None) -> None:
#     tif_dir = nc_dir.parent
#     tif_paths = return_list_filepaths(tif_dir, ".tif")
#     for tif_path in tqdm(
#         tif_paths, total=len(tif_paths), desc="Writing tifs to nc files"
#     ):
#         # filename = str(file_ops.get_n_last_subparts_path(tif, 1))
#         filename = tif_path.stem
#         tif_array = tif_to_xa_array(tif_path)
#         # xa_array_dict[filename] = tif_array.rename(filename)
#         if target_resolution_d:
#             tif_array = spatial_data.upsample_xarray_to_target(
#                 xa_array=tif_array, target_resolution=target_resolution_d
#             )
#         # save array to nc file
#         save_nc(tif_dir, filename, tif_array)

#     return tif_paths
#     print(f"All tifs converted to xarrays and stored as .nc files in {nc_dir}.")

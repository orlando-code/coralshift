from __future__ import annotations

import xarray as xa
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_gradient_magnitude

from coralshift.utils import file_ops, directories
from coralshift.processing import spatial_data

# these two imports necessary for importing US coastal bathymetry data
import requests
from bs4 import BeautifulSoup


class ReefAreas:
    name_mapping = {
        "A": "Great_Barrier_Reef_A_2020_30m",
        "B": "Great_Barrier_Reef_B_2020_30m",
        "C": "Great_Barrier_Reef_C_2020_30m",
        "D": "Great_Barrier_Reef_D_2020_30m",
        # Add more mappings as needed
    }

    def __init__(self):
        self.datasets = [
            {
                "name": "Great_Barrier_Reef_A_2020_30m",
                "short_name": "A",
                "file_name": "Great_Barrier_Reef_A_2020_30m_MSL_cog.tif",
                "short_file_name": "Great_Barrier_Reef_A",
                "xarray_name": "bathymetry_A",
                "lat_range": (-10, -17),
                "lon_range": (142, 147),
                "url": "https://ausseabed-public-warehouse-bathymetry.s3.ap-southeast-2.amazonaws.com/L3/0b9ad3f3-7ade-40a7-ae70-f7c6c0f3ae2e/Great_Barrier_Reef_A_2020_30m_MSL_cog.tif",  # noqa
            },
            {
                "name": "Great_Barrier_Reef B 2020 30m",
                "short_name": "B",
                "file_name": "Great_Barrier_Reef_B_2020_30m_MSL_cog.tif",
                "short_file_name": "Great_Barrier_Reef_B",
                "xarray_name": "bathymetry_B",
                "lat_range": (-16, -23),
                "lon_range": (144, 149),
                "url": "https://ausseabed-public-warehouse-bathymetry.s3.ap-southeast-2.amazonaws.com/L3/4a6e7365-d7b1-45f9-a576-2be8ff8cd755/Great_Barrier_Reef_B_2020_30m_MSL_cog.tif",  # noqa
            },
            {
                "name": "Great_Barrier_Reef C 2020 30m",
                "short_name": "C",
                "file_name": "Great_Barrier_Reef_C_2020_30m_MSL_cog.tif",
                "short_file_name": "Great_Barrier_Reef_C",
                "xarray_name": "bathymetry_C",
                "lat_range": (-18, -24),
                "lon_range": (148, 154),
                "url": "https://ausseabed-public-warehouse-bathymetry.s3.ap-southeast-2.amazonaws.com/L3/3b171f8d-9248-4aeb-8b32-0737babba3c2/Great_Barrier_Reef_C_2020_30m_MSL_cog.tif",  # noqa
            },
            {
                "name": "Great_Barrier_Reef D 2020 30m",
                "short_name": "D",
                "file_name": "Great_Barrier_Reef_D_2020_30m_MSL_cog.tif",
                "short_file_name": "Great_Barrier_Reef_D",
                "xarray_name": "bathymetry_D",
                "lat_range": (-23, -29),
                "lon_range": (150, 156),
                "url": "https://ausseabed-public-warehouse-bathymetry.s3.ap-southeast-2.amazonaws.com/L3/7168f130-f903-4f2b-948b-78508aad8020/Great_Barrier_Reef_D_2020_30m_MSL_cog.tif",  # noqa
            },
        ]

    def get_long_name_from_short(self, short_name):
        return self.name_mapping.get(short_name, "Unknown")

    def get_name_from_names(self, name):
        for dataset in self.datasets:
            if (
                dataset["name"] == name
                or dataset["short_name"] == name
                or dataset["file_name"] == name
            ):
                return dataset["name"]
        raise ValueError(f"'{name}' not a dataset.")

    def get_filename(self, name):
        name = self.get_name_from_names(name)
        dataset = self.get_dataset(name)
        if dataset:
            return dataset["file_name"]
        return None

    def get_short_filename(self, name):
        name = self.get_name_from_names(name)
        dataset = self.get_dataset(name)
        if dataset:
            return dataset["short_file_name"]
        return None

    def get_xarray_name(self, name):
        name = self.get_name_from_names(name)
        dataset = self.get_dataset(name)
        if dataset:
            return dataset["xarray_name"]
        return None

    def get_dataset(self, name):
        name = self.get_name_from_names(name)
        for dataset in self.datasets:
            if dataset["name"] == name:
                return dataset
        return None

    def get_lat_lon_limits(self, name):
        dataset = self.get_dataset(name)
        if dataset:
            return dataset["lat_range"], dataset["lon_range"]
        return None

    def get_url(self, name):
        name = self.get_name_from_names(name)
        dataset = self.get_dataset(name)
        if dataset:
            return dataset["url"]
        return None


def ensure_bathymetry_downloaded(area_name: str, loading_bar: bool = True) -> ReefAreas:
    """
    Ensures that the bathymetry data for the specified area is downloaded.

    Parameters
    ----------
        area_name (str): The name of the area.
        loading_bar (bool, optional): Whether to display a loading bar during the download process.
                                      Defaults to True.

    Returns
    -------
        ReefAreas: An instance of the ReefAreas class representing information about the downloaded area, and the other
        potentials.
    """
    areas_info = ReefAreas()

    # get url
    area_url = areas_info.get_url(area_name)
    # generate path to save data to
    save_path = directories.get_bathymetry_datasets_dir() / areas_info.get_filename(
        area_name
    )
    # download data if not already there
    file_ops.check_exists_download_url(save_path, area_url)

    return ReefAreas()


def calculate_gradient_magnitude(xa_da: xa.DataArray, sigma: int = 1):
    """
    Calculate the gradient magnitude of a DataArray using Gaussian gradient magnitude.

    Parameters
    ----------
        xa_da (xarray.DataArray): The input DataArray.
        sigma (int, optional): Standard deviation of the Gaussian filter. Default is 1.

    Returns
    -------
        xa.DataArray: The gradient magnitude of the input DataArray.
    """
    # Calculate the gradient magnitude using gaussian_gradient_magnitude. Sigma specifies kernel size.
    gradient_magnitude = gaussian_gradient_magnitude(xa_da.compute(), sigma=sigma)

    return xa.DataArray(
        gradient_magnitude, coords=xa_da.coords, dims=xa_da.dims, attrs=xa_da.attrs
    ).chunk(chunks="auto")


def generate_gradient_magnitude_nc(xa_da: xa.DataArray, sigma: int = 1):
    """
    Generate a NetCDF file containing the gradient magnitude of a DataArray. The NetCDF file is saved in the directory
    specified by directories.get_gradients_dir(), with the filename format
    "{data_array_name}_{spatial_resolution}_gradients.nc".

    Parameters
    ----------
        xa_da (xarray.DataArray): The input DataArray.
        sigma (int, optional): Standard deviation of the Gaussian filter. Default is 1.

    Returns
    -------
        tuple[xa.DataArray, str]: A tuple containing the gradient DataArray and the path to the saved NetCDF file.
    """
    # generate savepath
    resolution_d = np.mean(spatial_data.calculate_spatial_resolution(xa_da))
    filename = f"{xa_da.name}_{resolution_d:.5f}_gradients.nc"
    save_path = directories.get_gradients_dir() / filename

    # generate/save file as necessary
    if not save_path.is_file():
        grad_da = calculate_gradient_magnitude(xa_da, sigma)
        grad_da.to_netcdf(save_path)
        print(f"{filename} saved at {save_path}.")
    else:
        print(f"{filename} already exists at {save_path}.")
        grad_da = xa.open_dataarray(save_path)

    return grad_da, save_path


def generate_bathymetry_xa_da(area_name: str):
    """
    Generate bathymetry data for a specified area.

    Parameters
    ----------
        area_name (str): The name of the area.

    Returns
    -------
        tuple[str, xa.DataArray]: A tuple containing the filepath and the processed xarray of the generated bathymetry
        data.
    """
    # download .tif if not downloaded aready
    reef_areas = ensure_bathymetry_downloaded(area_name)
    # cast tif to processed xarray with correct crs
    xa_bath = spatial_data.tif_to_xarray(
        directories.get_bathymetry_datasets_dir() / reef_areas.get_filename(area_name),
        reef_areas.get_xarray_name(area_name),
    )

    resolution = np.mean(spatial_data.calculate_spatial_resolution(xa_bath))

    bath_name = f"{reef_areas.get_xarray_name(area_name)}_{resolution:.05f}d"
    filepath, xa_da = file_ops.save_nc(
        directories.get_bathymetry_datasets_dir(), bath_name, xa_bath, return_array=True
    )

    return filepath, xa_da


######################################################
# Not used within MRes: coastal bathymetry for US only
######################################################


def fetch_links_from_url(page_url: str, suffix: str = None) -> list[str]:
    """Fetches all links (as href attributes) from a webpage and returns a list of them. If a `suffix` argument is
    provided, only the links that end with that suffix are included.

    Parameters
    ----------
        page_url (str): The URL of the webpage to fetch links from.
        suffix (str, optional): The suffix to filter links by. Defaults to None.

    Returns
    -------
        list[str]: A list of links from the webpage.
    """
    reqs = requests.get(page_url)
    soup = BeautifulSoup(reqs.text, "html.parser")
    link_segments = soup.find_all("a")
    # extract link strings, excluding None values
    links = [link.get("href") for link in link_segments if link.get("href") is not None]

    if suffix:
        links = [link for link in links if link.endswith(file_ops.pad_suffix(suffix))]

    return links


def download_etopo_data(
    download_dest_dir: Path | str,
    resolution: str | int = 15,
    file_extension: str = ".nc",
    loading_bar: bool = True,
) -> None:
    """
    Downloads ETOPO data files from the NOAA website and saves them to the specified directory.

    Parameters
    ----------
        download_dest_dir (Path | str): The directory to save the downloaded files to.
        resolution (str | int, optional): The resolution of the data in degrees (15, 30, 60). Defaults to 15.
        file_extension (str, optional): The file extension to filter downloads by. Defaults to '.nc'.
        loading_bar (bool, optional): Whether to display a progress bar during download. Defaults to True.

    Returns
    -------
        None
    """
    file_server_url = "https://www.ngdc.noaa.gov/thredds/fileServer/global/"
    page_url = f"https://www.ngdc.noaa.gov/thredds/catalog/global/ETOPO2022/{resolution}s/{resolution}s_geoid_netcdf/catalog.html"  # noqa

    for link in fetch_links_from_url(page_url, file_extension):
        # file url involves multiple levels
        file_specifier = file_ops.get_n_last_subparts_path(Path(link), 4)
        file_name = file_ops.get_n_last_subparts_path(Path(link), 1)

        url = file_server_url + str(file_specifier)
        download_dest_path = Path(download_dest_dir, file_name)

        file_ops.check_exists_download_url(download_dest_path, url, loading_bar)

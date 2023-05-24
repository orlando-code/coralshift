from __future__ import annotations

import requests
from bs4 import BeautifulSoup
from pathlib import Path
from coralshift.utils import file_ops


# def download_etopo_data(download_dest_dir: Path | str, resolution: str | int, loading_bar: bool = True) -> None:
#     """Download bathymetry data from the NOAA ETOPO Global Relief Model dataset. This may be overspecific and links
#     could change. Currently stored at: https://www.ncei.noaa.gov/products/etopo-global-relief-model

#     Parameters
#     ----------
#         download_dest_dir (Path | str): destination directory where the downloaded file will be saved.
#         resolution (str | int): desired resolution of the data. Can be one of the following: 15, 30, or 60.
#         loading_bar (bool, optional): If True, display a progress bar while downloading the file. Default is True.

#     Returns
#     -------
#         None
#     """

#     bathymetry_url_page = f"https://www.ngdc.noaa.gov/thredds/catalog/global/ETOPO2022/{resolution}s/{resolution}s_geoid_netcdf/catalog.html" # noqa
#     file_server_url = "https://www.ngdc.noaa.gov/thredds/fileServer/global/"

#     reqs = requests.get(bathymetry_url_page)
#     soup = BeautifulSoup(reqs.text, 'html.parser')

#     # traverse paragraphs from soup
#     for link in soup.find_all("a"):
#         # if ends with nc
#         if file_ops.check_path_suffix(link.get("href"), "nc"):
#             # file url involves multiple levels
#             file_specifier = file_ops.get_n_last_subparts_path(Path(link.get("href")), 4)
#             file_name = file_ops.get_n_last_subparts_path(Path(link.get("href")), 1)

#             url = file_server_url + str(file_specifier)
#             download_dest_path = Path(download_dest_dir, file_name)

#             file_ops.check_exists_download_url(download_dest_path, url, loading_bar)


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


def download_30m_gbr_bathymetry(
    download_dest_dir: Path | str, areas: list[str] = ["A", "B", "C", "D"]
) -> None:
    """Download bathymetry data for the Great Barrier Reef (GBR) region in TIFF format from AWS S3 bucket.
    Dataset DOI: 10.4225/25/5a207b36022d2

    Parameters
    ----------
        download_dest_dir (Path | str): Path to the directory where the downloaded files should be saved.
        areas (list[str]): A list of strings indicating the areas to be downloaded. The possible values are
            ['A', 'B', 'C', 'D'], corresponding to four different parts of the GBR region. Defaults to download all
            these areas.

    Returns
    -------
        None
    """
    page_url = "https://researchdata.edu.au/high-resolution-depth-30-m/1278835"
    data_links = fetch_links_from_url(page_url, ".tif")

    for area_url in list(data_links):
        area_filename = file_ops.get_n_last_subparts_path(area_url, 1)
        filepath = Path(download_dest_dir, area_filename)
        # check whether file already downloaded: if not, download
        file_ops.check_exists_download_url(filepath, area_url)

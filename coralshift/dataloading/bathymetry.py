import requests
from bs4 import BeautifulSoup
from pathlib import Path
from coralshift.utils import file_ops


def download_etopo_data(download_dest_dir: Path | str, resolution: str | int, loading_bar: bool = True) -> None:
    """Download bathymetry data from the NOAA ETOPO Global Relief Model dataset. This may be overspecific and links
    could change. Currently stored at: https://www.ncei.noaa.gov/products/etopo-global-relief-model

    Parameters
    ----------
        download_dest_dir (Path | str): destination directory where the downloaded file will be saved.
        resolution (str | int): desired resolution of the data. Can be one of the following: 15, 30, or 60.
        loading_bar (bool, optional): If True, display a progress bar while downloading the file. Default is True.

    Returns
    -------
        None
    """

    bathymetry_url_page = f"https://www.ngdc.noaa.gov/thredds/catalog/global/ETOPO2022/{resolution}s/{resolution}s_geoid_netcdf/catalog.html" # noqa
    file_server_url = "https://www.ngdc.noaa.gov/thredds/fileServer/global/"

    reqs = requests.get(bathymetry_url_page)
    soup = BeautifulSoup(reqs.text, 'html.parser')

    # traverse paragraphs from soup
    for link in soup.find_all("a"):
        # if ends with nc
        if file_ops.check_path_suffix(link.get("href"), "nc"):
            # file url involves multiple levels
            file_specifier = file_ops.get_n_last_subparts_path(Path(link.get("href")), 4)
            file_name = file_ops.get_n_last_subparts_path(Path(link.get("href")), 1)

            url = file_server_url + str(file_specifier)
            download_dest_path = Path(download_dest_dir, file_name)

            file_ops.check_exists_download_url(download_dest_path, url, loading_bar)


def download_files_via_bs4(page_url: str, file_extension: str):
# TODO: sort this download
    reqs = requests.get(page_url)
    soup = BeautifulSoup(reqs.text, 'html.parser')

    for link in soup.find_all("a"):
        # if ends with "file_extension"
        if file_ops.check_path_suffix(link.get("href"), file_extension):
            # file url involves multiple levels
            file_specifier = file_ops.get_n_last_subparts_path(Path(link.get("href")), 4)
            file_name = file_ops.get_n_last_subparts_path(Path(link.get("href")), 1)

            url = file_server_url + str(file_specifier)
            download_dest_path = Path(download_dest_dir, file_name)

            file_ops. check_exists_download_url(download_dest_path, url, loading_bar)


def download_30m_gbr_bathymetry(download_dest_dir: Path | str, areas: list[str]) -> None:
    """Download bathymetry data for the Great Barrier Reef (GBR) region in TIFF format from AWS S3 bucket. This is VERY
    hacky: but couldn't figure out how to scrape links from buttons: there seems to be no html which differentiates one
    button from another... TODO: use selenium to interact with page. Not a priority.
    Dataset DOI: 10.4225/25/5a207b36022d2

    Parameters
    ----------
        download_dest_dir (Path | str): Path to the directory where the downloaded files should be saved.
        areas (list[str]): A list of strings indicating the areas to be downloaded. The possible values are
            ['A', 'B', 'C', 'D'], corresponding to four different parts of the GBR region.

    Returns
    -------
        None
    """

    bathymetry_url_page = "https://researchdata.edu.au/high-resolution-depth-30-m/1278835"



    data_urls = ["https://ausseabed-public-warehouse-bathymetry.s3.ap-southeast-2.amazonaws.com/L3/0b9ad3f3-7ade-40a7-ae70-f7c6c0f3ae2e/Great_Barrier_Reef_A_2020_30m_MSL_cog.tif", # noqa
        "https://ausseabed-public-warehouse-bathymetry.s3.ap-southeast-2.amazonaws.com/L3/4a6e7365-d7b1-45f9-a576-2be8ff8cd755/Great_Barrier_Reef_B_2020_30m_MSL_cog.tif", # noqa
        "https://ausseabed-public-warehouse-bathymetry.s3.ap-southeast-2.amazonaws.com/L3/3b171f8d-9248-4aeb-8b32-0737babba3c2/Great_Barrier_Reef_C_2020_30m_MSL_cog.tif", # noqa
        "https://ausseabed-public-warehouse-bathymetry.s3.ap-southeast-2.amazonaws.com/L3/7168f130-f903-4f2b-948b-78508aad8020/Great_Barrier_Reef_D_2020_30m_MSL_cog.tif"] # noqa

    for area_url in list(data_urls):
        area_filename = file_ops.get_n_last_subparts_path(area_url, 1)
        # area_filename = f"Great_Barrier_Reef_{alpha.upper()}_2020_30m_MSL_cog.tif"
        # area_url = '/'.join((start_data_url, area_filename))

        filepath = Path(download_dest_dir, area_filename)
        # check whether file already downloaded
        file_ops.check_exists_download_url(filepath, area_url)

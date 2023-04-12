import requests
from bs4 import BeautifulSoup
from pathlib import Path
from coralshift.utils import file_ops


def download_etopo_data(download_dest_dir: Path | str, resolution: str | int, loading_bar: bool = True) -> None:
    """Download bathymetry data from the NOAA ETOPO Global Relief Model dataset.

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

            file_ops. check_exists_download_url(download_dest_path, url, loading_bar)

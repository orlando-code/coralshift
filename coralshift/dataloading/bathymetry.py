# general
import numpy as np

# spatial
import xarray as xa
from scipy.ndimage import gaussian_gradient_magnitude

# custom
from coralshift.utils import config
from coralshift.processing import spatial_data

# from coralshift.cmipper import file_ops as cmipper_file_ops
from cmipper import file_ops as cmipper_file_ops


def generate_gebco_xarray(
    lats: list[float, float] = [-40, 0],
    lons: list[float, float] = [130, 170],
):
    # TODO: make generic for other bathymetry data?
    # TODO: mosaicing of files
    gebco_xa_dir = config.bathymetry_dir / "gebco"
    gebco_xa_dir.mkdir(parents=True, exist_ok=True)

    # check if there exists a file spanning an adequate spatial extent
    nc_fps = list(gebco_xa_dir.glob("*.nc"))
    subset_fps = cmipper_file_ops.find_files_for_area(nc_fps, lats, lons)
    if len(subset_fps) > 0:
        # arbitrarily select first suitable file
        gebco_fp = subset_fps[0]
    else:
        raise FileNotFoundError(
            """No GEBCO files with suitable geographic range found. Ensure necessary region is downloaded.
            Visit: https://download.gebco.net/"""
        )
    print(
        f"Loading gebco elevation xarray across {lats} latitudes & {lons} longitudes from {gebco_fp}."
    )
    return spatial_data.process_xa_d(xa.open_dataset(gebco_fp).sel(
        lat=slice(min(lats), max(lats)), lon=slice(min(lons), max(lons))
    ).drop_vars("crs").astype(np.float64))  # TODO: proper chunking
    # this is hacky to drop crs and slice before processing, but this array is huge and crs (str)
    # interferes with thresholding data


def generate_gebco_slopes_xarray(
    lats: list[float, float] = [-40, 0],
    lons: list[float, float] = [130, 170],
):
    gebco_xa_dir = config.bathymetry_dir / "gebco/rasters"
    gebco_xa_dir.mkdir(parents=True, exist_ok=True)

    # check if there exists a file spanning an adequate spatial extent
    # TODO: ideally need to buffer to allow calculation of edge cases
    nc_fps = list(gebco_xa_dir.glob("*.nc"))
    subset_fps = cmipper_file_ops.find_files_for_area(nc_fps, lats, lons)
    # select fps from subset_fps which contain "slope"
    subset_slope_fps = [fp for fp in subset_fps if "slope" in str(fp)]

    if len(subset_slope_fps) > 0:
        # arbitrarily select first suitable file
        print(
            f"Loading seafloor slopes xarray across {lats} latitudes & {lons} longitudes from {subset_slope_fps[0]}."
        )
        return xa.open_dataset(subset_slope_fps[0]).sel(
            latitude=slice(min(lats), max(lats)), longitude=slice(min(lons), max(lons))
        )
    else:
        gebco_xa = generate_gebco_xarray(lats, lons)
        print("calculating slopes from bathymetry...")
        return (
            spatial_data.process_xa_d(
                calculate_gradient_magnitude(gebco_xa["elevation"])
            )
            .to_dataset(name="slope")
            .sel(
                latitude=slice(min(lats), max(lats)),
                longitude=slice(min(lons), max(lons)),
            )
        )


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

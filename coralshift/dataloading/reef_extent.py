import json
import pandas as pd
import geopandas as gpd
# import numpy as np
# import rasterio
import xarray as xa
from tqdm import tqdm

from coralshift.processing import spatial_data
from coralshift.utils import directories, file_ops, utils


def generate_area_geojson(area_class, area_name: str) -> None:
    """
    Generate a GeoJSON file representing a specific area.

    Parameters
    ----------
        area_class (ReefAreas): An instance of the a class containing area information.
        area_name (str): The name of the area for which to generate the GeoJSON.
        save_dir (Path or str): The directory path where the GeoJSON file will be saved.

    Returns
    -------
        output_path (Path): Path to GeoJSON file.
    """
    # Save the GeoJSON data to a file
    name = area_class.get_short_filename(area_name)
    save_dir = file_ops.guarantee_existence(directories.get_reef_baseline_dir() / name)

    filename = f"{name}.geojson"
    output_path = save_dir / filename

    if not output_path.is_file():
        lat_range, lon_range = area_class.get_lat_lon_limits(area_name)
        # create a geojson object
        geojson_data = generate_area_geojson_info(
            lat_range=lat_range, lon_range=lon_range, name=name
        )

        with open(output_path, "w") as file:
            json.dump(geojson_data, file)
    else:
        print(f"File already exists at {output_path}. Check if this is correct.")

    return output_path


def generate_area_geojson_info(
    lat_range: tuple[float], lon_range: tuple[float], name: str = "placeholder"
) -> dict:
    """Generate a GeoJSON feature collection representing a rectangular area.

    Parameters
    ----------
        lat_range (tuple[float]): A tuple containing the latitude range (min, max) of the area.
        lon_range (tuple[float]): A tuple containing the longitude range (min, max) of the area.
        name (str, optional): Name or identifier for the area (default is "placeholder").

    Returns
    -------
        dict: A GeoJSON object representing the rectangular area in the form of a Feature Collection.

    Example:
        # Generate a GeoJSON object representing an area from latitude 10 to 20 and longitude 30 to 40
        geojson_info = generate_area_geojson_info((10, 20), (30, 40), name="Sample Area")
    """
    features = [
        {
            "type": "Feature",
            "geometry": {
                "type": "MultiPolygon",
                "coordinates": [
                    [
                        [
                            [lon_range[0], lat_range[0]],
                            [lon_range[1], lat_range[0]],
                            [lon_range[1], lat_range[1]],
                            [lon_range[0], lat_range[1]],
                            [lon_range[0], lat_range[0]],
                        ]
                    ]
                ],
            },
            "properties": {"name": name, "format": "GeoJSON"},
        }
    ]

    # return GeoJSON object
    return {"type": "FeatureCollection", "features": features}


def process_benthic_pd(
    benthic_df: pd.DataFrame,
    limit_to: list[str] = ["Coral/Algae"],
    geometry_col: str = "geometry",
) -> pd.DataFrame:
    # don't overwite original values
    working_df = benthic_df.copy()

    # define Coral/Algae as target
    class_vals = {
        "Coral/Algae": 1,
        "Reef": 2,
        "Rock": 3,
        "Rubble": 4,
        "Sand": 5,
        "Microalgal Mats": 6,
        "Seagrass": 7,
    }
    working_df["class_val"] = working_df["class"].map(class_vals)

    # Convert the values in "class_val" column to integers
    working_df["class_val"] = working_df["class_val"].astype(int)

    # Filter the DataFrame to include only the specified classes
    filtered_df = working_df[working_df["class"].isin(limit_to)]
    return gpd.GeoDataFrame(filtered_df, geometry=geometry_col)


# def rasterize_gdf(gdf: gpd.GeoDataFrame, chunk_size: int = 100):
#     """Rasterizes a GeoDataFrame into a raster array.
#     N.B. prohibitively computationally-expensive to run locally

#     Parameters
#     ----------
#         gdf (gpd.GeoDataFrame): The GeoDataFrame containing the geometries to rasterize.
#         chunk_size (int, optional): The size of the chunks to process the GeoDataFrame. Defaults to 100.

#     Returns
#     -------
#         np.ndarray: The rasterized array representing the GeoDataFrame.
#     """

#     # Prepare raster parameters
#     lon_min, lat_min, lon_max, lat_max = gdf.total_bounds
#     (lat_distance, lon_distance) = spatial_data.degrees_to_distances(
#         lat_max - lat_min,
#         lon_max - lon_min,
#         np.mean((lat_min, lat_max)),
#         np.mean((lon_max, lon_min)),
#     )
#     width = int(lon_distance)
#     height = int(lat_distance)
#     pixel_size = 5
#     transform = rasterio.transform.from_origin(
#         lon_min, lat_max, xsize=pixel_size, ysize=pixel_size
#     )

#     # Create an empty raster array
#     raster_array = np.zeros((height, width), dtype=np.uint8)  # Use the desired dtype

#     # Process the geodataframe in chunks
#     for i in tqdm(range(0, len(gdf), chunk_size), desc="Rasterizing chunks"):
#         # Get the chunk of data
#         chunk = gdf.iloc[i : i + chunk_size]  # noqa

#         # Get the shapes and values for the chunk
#         shapes = (
#             (geom, value) for geom, value in zip(chunk.geometry, chunk["class_val"])
#         )

#         # Rasterize the shapes in the chunk
#         out = rasterio.rasterize(shapes=shapes, out=raster_array, transform=transform)

#     return out


def process_coral_gt_tifs(tif_dir_name=None, target_resolution_d: float = None):
    if not tif_dir_name:
        tif_dir = directories.get_reef_baseline_dir()
    else:
        tif_dir = directories.get_reef_baseline_dir() / tif_dir_name

    nc_dir = file_ops.guarantee_existence(tif_dir / "gt_nc_dir")
    # save tifs to ncs in new dir
    tif_paths = file_ops.tifs_to_ncs(nc_dir, target_resolution_d)
    # get list of nc paths in dir
    xa_arrays_list = [file_ops.tif_to_xa_array(tif_path) for tif_path in tif_paths]
    # merge ncs into one mega nc file
    if len(xa_arrays_list) > 1:
        concatted = xa.concat(xa_arrays_list, dim=["latitude", "longitude"])
    else:
        concatted = xa_arrays_list[0]
    file_ops.save_nc(
        nc_dir, f"concatenated_{target_resolution_d:.05f}_degree", concatted
    )


def generate_coral_shp(gdf_coral: gpd.GeoDataFrame, file_name: str) -> None:
    save_dir = directories.get_reef_baseline_dir() / file_name
    save_path = save_dir / f"{file_name}_coral_gt.shp"

    if not save_path.is_file():
        gdf_coral.to_file(save_path, driver="ESRI Shapefile")
    else:
        print(f"File at {save_path} already exists.")


def process_reef_extent_tifs(target_resolution_d: float = None):
    # fetch list of reef presence tifs
    gt_tif_files = file_ops.return_list_filepaths(
        directories.get_reef_baseline_dir(), ".tif", incl_subdirs=False
    )
    # generate dictionary of file names and arrays: {filename: xarray.DataArray, ...}
    gt_nc_dict = spatial_data.tifs_to_xa_array_dict(gt_tif_files)

    resampled_gt_nc_dict = {}
    res_string = utils.generate_resolution_str(resolution_d=target_resolution_d, sfs=4)
    for name, xa_d in tqdm(
        gt_nc_dict.items(), desc=f" Resampling xarray objects to {res_string}"
    ):
        new_name = f"{file_ops.remove_suffix(name)}_rs_{res_string}"
        if target_resolution_d:
            xa_d = spatial_data.resample_xarray_to_target(
                xa_d, target_resolution_d=target_resolution_d, name=new_name
            )
        resampled_gt_nc_dict[new_name] = xa_d
    # save dictionary of tifs to nc, if files not already existing
    file_ops.save_dict_xa_ds_to_nc(gt_nc_dict, directories.get_gt_files_dir())

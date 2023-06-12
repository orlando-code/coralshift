import json
import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio

from pathlib import Path
from tqdm import tqdm

from coralshift.processing import spatial_data


def generate_area_geojson(area_class, area_name: str, save_dir: Path | str) -> None:
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
    name = area_class.get_name_from_names(area_name)

    filename = f"{name}.geojson"
    output_path = save_dir / filename

    if not output_path.is_file():
        lat_range, lon_range = area_class.get_lat_lon_limits(area_name)

        # Create a GeoJSON feature collection
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

        # Create a GeoJSON object
        geojson_data = {"type": "FeatureCollection", "features": features}

        with open(output_path, "w") as file:
            json.dump(geojson_data, file)
    else:
        print(f"File already exists at {output_path}. Check if this is correct.")

    return output_path


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


def rasterize_gdf(gdf: gpd.GeoDataFrame, chunk_size: int = 100):
    """Rasterizes a GeoDataFrame into a raster array.
    N.B. prohibitively computationally-expensive to run locally

    Parameters
    ----------
        gdf (gpd.GeoDataFrame): The GeoDataFrame containing the geometries to rasterize.
        chunk_size (int, optional): The size of the chunks to process the GeoDataFrame. Defaults to 100.

    Returns
    -------
        np.ndarray: The rasterized array representing the GeoDataFrame.
    """

    # Prepare raster parameters
    lon_min, lat_min, lon_max, lat_max = gdf.total_bounds
    (lat_distance, lon_distance) = spatial_data.degrees_to_distances(
        lat_max - lat_min,
        lon_max - lon_min,
        np.mean((lat_min, lat_max)),
        np.mean((lon_max, lon_min)),
    )
    width = int(lon_distance)
    height = int(lat_distance)
    pixel_size = 5
    transform = rasterio.transform.from_origin(
        lon_min, lat_max, xsize=pixel_size, ysize=pixel_size
    )

    # Create an empty raster array
    raster_array = np.zeros((height, width), dtype=np.uint8)  # Use the desired dtype

    # Process the geodataframe in chunks
    for i in tqdm(range(0, len(gdf), chunk_size), desc="Rasterizing chunks"):
        # Get the chunk of data
        chunk = gdf.iloc[i : i + chunk_size]  # noqa

        # Get the shapes and values for the chunk
        shapes = (
            (geom, value) for geom, value in zip(chunk.geometry, chunk["class_val"])
        )

        # Rasterize the shapes in the chunk
        out = rasterio.rasterize(shapes=shapes, out=raster_array, transform=transform)

    return out

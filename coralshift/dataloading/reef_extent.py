import json
import pandas as pd
import geopandas as gpd
from pathlib import Path


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

    lat_range, lon_range = area_class.get_lat_lon_limits(area_name)
    name = area_class.get_name_from_names(area_name)

    # Create a GeoJSON feature collection
    features = [
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [lon_range[0], lat_range[0]],
                        [lon_range[1], lat_range[0]],
                        [lon_range[1], lat_range[1]],
                        [lon_range[0], lat_range[1]],
                        [lon_range[0], lat_range[0]],
                    ]
                ],
            },
            "properties": {"name": name, "format": "GeoJSON"},
        }
    ]

    # Create a GeoJSON object
    geojson_data = {"type": "FeatureCollection", "features": features}

    # Save the GeoJSON data to a file
    filename = f"{name}.geojson"

    output_path = save_dir / filename
    with open(output_path, "w") as file:
        json.dump(geojson_data, file)

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

# TODO: automate with nice data configs of selected variables etc.
# TODO: logging functions

import xarray as xa
from coralshift.dataloading import config
from coralshift import functions_creche
from coralshift.processing import spatial_data
from coralshift.machine_learning import baselines
from coralshift import functions_creche
from pathlib import Path
import geopandas as gpd
import dask_geopandas as daskgpd

import dask.dataframe as dd
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(df, label_column):
    # Split dataframe into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df.loc[:, df.columns!=label_column], df[label_column], test_size=0.2, random_state=42)
    
    # Min-max scaling ignoring NaNs
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train.drop(columns=[label_column]))
    X_test_scaled = scaler.transform(X_test.drop(columns=[label_column]))
    
    # Convert scaled arrays back to Dask DataFrame
    columns = X_train.drop(columns=[label_column]).columns
    X_train_scaled_df = dd.from_dask_array(pd.DataFrame(X_train_scaled, columns=columns))
    X_test_scaled_df = dd.from_dask_array(pd.DataFrame(X_test_scaled, columns=columns))
    
    # Generate a column showing 1 if any NaN is in the row, 0 otherwise
    X_train_scaled_df['onehot_nan'] = X_train_scaled_df.isnull().any(axis=1).astype(int)
    X_test_scaled_df['onehot_nan'] = X_test_scaled_df.isnull().any(axis=1).astype(int)
    
    # Replace NaNs with 0
    X_train_scaled_df = X_train_scaled_df.fillna(0)
    X_test_scaled_df = X_test_scaled_df.fillna(0)
    
    return X_train_scaled_df, X_test_scaled_df




data_fp = config.data_folder

### USER INPUTS ###
# resolution_lat, resolution_lon = 1, 1
lats = [-34, 0]
# lats = [-25, 0]
lons = [130, 170]


def get_data(
    model_type: str,
    resolution_lat: float,
    resolution_lon: float,
    train_val_test_frac: list = [0.7, 0.15, 0.15],
    gt: str = "unep_coral_presence",
    exclude_list: list = [],
):
    ### DATA ###
    print("loading cmip data...")
    # buffered cmip6 xa.Dataset
    cmip6_fp = (Path(config.cmip6_data_folder)
        / "BCC-CSM2-HR/r1i1p1f1/uo_so_vo_thetao_tos_buffered.nc"
    )
    cmip6_xa = xa.open_dataset(cmip6_fp, chunks="auto")

    print("loading gebco data...")
    # gebco bathymetry fp
    gebco_fp = (Path(config.bathymetry_folder) / "gebco/gebco_2023_n0.0_s-40.0_w130.0_e170.0.nc"
    )
    gebco_xa = xa.open_dataset(gebco_fp, chunks="auto")

    # gebco slopes fp
    gebco_slopes_fp = (Path(config.bathymetry_folder)
        / "gebco/gebco_2023_n0.0_s-40.0_w130.0_e170.0_slopes.nc"
    )
    gebco_slopes_xa = xa.open_dataset(gebco_slopes_fp, chunks="auto")

    print("loading unep-wcmc data...")
    # unep_wcmc shp fp
    unep_fp = (Path(config.gt_folder) / "unep_wcmc/01_Data/WCMC008_CoralReef2021_Py_v4_1.shp"
    )
    unep_gdf = daskgpd.read_file(unep_fp, npartitions=4)
    # generate gt raster
    unep_raster = functions_creche.rasterize_geodf(
        unep_gdf, resolution_lat=resolution_lat, resolution_lon=resolution_lon
    )
    # generate gt xarray
    unep_xa = functions_creche.raster_to_xarray(
        unep_raster,
        x_y_limits=functions_creche.lat_lon_vals_from_geo_df(unep_gdf)[:4],
        resolution_lat=resolution_lat,
        resolution_lon=resolution_lon,
        name="unep_coral_presence",
    ).chunk("auto")
    
    ### PREPROCESSING ###

    # derive
    if model_type == "rf":
        print("calculating statistics for rf model...")
        # compute stats df
        cmip6_xa = functions_creche.calculate_statistics(cmip6_xa)

    print("spatially aligning datasets...")
    # spatially align datasets into a single xarray dataset
    input_dss = [
        spatial_data.process_xa_d(xa_d)
        for xa_d in [cmip6_xa, gebco_xa, gebco_slopes_xa, unep_xa]
    ]
    common_dataset = functions_creche.spatially_combine_xa_d_list(
        input_dss, lats, lons, resolution_lat, resolution_lon
    )
    
    common_dataset.to_netcdf(Path(data_fp) / f"temp_{model_type}_{functions_creche.tuples_to_string(lats, lons)}_ds.nc")

#     if model_type == "rf":
#         print("generating rf train/test arrays...")
#         common_df = common_dataset.to_dask_dataframe()
#         exclude_list += ["crs", "depth", "spatial_ref"]
#         predictors = [
#             pred
#             for pred in common_df.columns
#             if pred != gt and pred not in exclude_list
#         ]
        
#         return common_df

#         X_train_scaled_df, X_test_scaled_df = preprocess_data(common_df, label_column=gt)
#         (
#             (X_train, y_train),
#             (_, _),
#             (X_test, y_test),
#         ), dfs_list = functions_creche.process_df_for_rfr(
#             common_df, predictors, gt, train_val_test_frac=train_val_test_frac
#         )

#     return ((X_train, y_train), (_,_), (X_test, y_test)), dfs_list

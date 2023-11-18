# TODO: automate with nice data configs of selected variables etc.
# TODO: logging functions

import xarray as xa
from coralshift.dataloading import config
from coralshift import functions_creche
from coralshift.machine_learning import baselines

data_fp = config.data_folder

### USER INPUTS ###
resolution_lat, resolution_lon = 1, 1
lats = [-34, 0]
lons = [130, 170]


train_val_test_frac = [0.7, 0.15, 0.15]
model_type = "rf"

gt = "unep_coral_presence"
exclude_list = []  # variables to exclude in prediction

### DATA ###
# buffered cmip6 xa.Dataset
cmip6_fp = (
    Path(config.cmip6_data_folder)
    / "BCC-CSM2-HR/r1i1p1f1/uo_so_vo_thetao_tos_buffered.nc"
)
cmip6_xa = xa.open_dataset(cmip6_fp)

# gebco bathymetry fp
gebco_fp = (
    Path(config.bathymetry_folder) / "gebco/gebco_2023_n0.0_s-40.0_w130.0_e170.0.nc"
)
gebco_xa = xa.open_dataset(gebco_fp)

# gebco slopes fp
gebco_slopes_fp = (
    Path(config.bathymetry_folder)
    / "gebco/gebco_2023_n0.0_s-40.0_w130.0_e170.0_slopes.nc"
)
gebco_slopes_xa = xa.open_dataset(gebco_slopes_fp)

# unep_wcmc shp fp
unep_fp = Path(config.gt_folder) / "unep_wcmc/01_Data/WCMC008_CoralReef2021_Py_v4_1.shp"
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
)


### PREPROCESSING ###

# derive
if model_type == "rf":
    # compute stats df
    cmip6_xa = functions_creche.calculate_statistics(cmip6_xa)

# spatially align datasets into a single xarray dataset
input_dss = [
    spatial_data.process_xa_d(xa_d)
    for xa_d in [cmip6_xa, gebco_xa, gebco_slopes_xa, unep_xa]
]
common_dataset = functions_creche.spatially_combine_xa_d_list(
    input_dss, lats, lons, resolution_lat, resolution_lon
)


if model_type == "rf":
    train_val_test_frac = [0.8, 0, 0.2]
    common_df = common_dataset.to_dataframe()
    exclude_list += ["crs", "depth", "spatial_ref"]
    predictors = [
        pred for pred in common_df.columns if pred != gt and pred not in exclude_list
    ]

    (
        (X_train, y_train),
        (_, _),
        (X_test, y_test),
    ), dfs_list = functions_creche.process_df_for_rfr(
        common_df, predictors, gt, train_val_test_frac=train_val_test_frac
    )

    # train_random_model
    random_model = baselines.train_tune(
        X_train,
        y_train,
        "rf_reg",
        n_iter=2,
        cv=2,
        name="first_random",
        search_type="random",
        n_jobs=-1,
        verbose=0,
    )

    print(random_model.best_estimator_)

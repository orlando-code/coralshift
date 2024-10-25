import multiprocessing
from functools import lru_cache
import time

from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler, FunctionTransformer
from sklearn.metrics import root_mean_squared_error

from coralshift.dataloading import config

import xgboost as xgb
import numpy as np
import xarray as xa
from pathlib import Path

# COMPUTE SETUP
FRAC_COMPUTE = 0.25
CV_FOLDS = 3
N_ITER = 10
EARLY_STOPPING_ROUNDS = 10

# DATA SETUP
STATIC_DATA_DIR_FP = Path(config.static_cmip6_data_folder)
DEPTH_MASK_LIMS = [-50, 0]
GROUND_TRUTH_NAME = "unep_coral_presence"
RES_STR = "0-01"
TTV_FRACTIONS = [0.7, 0.2, 0.1]
SPLIT_METHOD = "pixelwise"
SCALE_METHOD_X = "minmax"
SCALE_METHOD_y = "log"
REMOVE_NAN_ROWS = True

DATA_SUBSET = 100


# if __name__ == "__main__":
def main():
    # DATA LOADING
    # load data. TODO: add multifile loading for machines with reasonable numbers of CPUs
    print("loading static data...")
    all_data_df = load_static_data(RES_STR)

    # DATA PREPROCESSING
    # apply depth mask
    print("preprocessing...")
    depth_masked_df = functions_creche.depth_filter(
        all_data_df, depth_mask_lims=DEPTH_MASK_LIMS
    )
    # split data into train, test, validation
    [train_df, test_df, val_df] = functions_creche.train_test_val_split(
        depth_masked_df, ttv_fractions=TTV_FRACTIONS, split_method=SPLIT_METHOD
    )
    # preprocess data: initialise scalers values
    scaler_X = generate_data_scaler(SCALE_METHOD_X)
    scaler_y = generate_data_scaler(SCALE_METHOD_y)
    # preprocess data: nan handling
    [train_df, test_df, val_df] = [
        onehot_nan(df, REMOVE_NAN_ROWS) for df in [train_df, test_df, val_df]
    ]
    # split into X and y for each subset
    (train_X, train_y), (test_X, test_y), (val_X, val_y) = [
        split_X_y(df, GROUND_TRUTH_NAME) for df in [train_df, test_df, val_df]
    ]
    # fit scalers
    scaler_X.fit(train_X)
    scaler_y.fit(train_y)
    # transform data
    (train_X, train_y), (test_X, test_y), (val_X, val_y) = [
        (scaler_X.transform(X), scaler_y.transform(y))
        for X, y in [(train_X, train_y), (test_X, test_y), (val_X, val_y)]
    ]
    # cast data to DMatrix for native xgboost handling
    dtrain, dtest, dval = [
        xgb.DMatrix(X, y)
        for X, y in [(train_X, train_y), (test_X, test_y), (val_X, val_y)]
    ]

    # MODEL FITTING
    print("fitting model...")
    sklearn_random_search(train_X[:DATA_SUBSET], train_y[:DATA_SUBSET])


if __name__ == "__main__":
    main()

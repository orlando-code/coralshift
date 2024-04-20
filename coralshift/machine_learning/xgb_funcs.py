import multiprocessing
from functools import lru_cache
import time

from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler, FunctionTransformer
from sklearn.metrics import root_mean_squared_error

from coralshift import functions_creche
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


# decorator to hold function output in memory
@lru_cache(maxsize=None)  # Set maxsize to limit the cache size, or None for unlimited
def load_static_data(res_str: str = "0-01"):
    # TODO: could replace with path to multifile dir, open with open_mfdataset and chunking
    high_res_ds_fp = STATIC_DATA_DIR_FP.glob(f"*_{res_str}_*.nc").__next__()
    high_res_ds = xa.open_dataset(high_res_ds_fp)
    all_data_df = high_res_ds.to_dataframe()

    return all_data_df


def split_X_y(df, target_name: str):
    y = df[target_name]
    X = df.drop(target_name, axis=1)

    return X, y


# def preprocess_data(X, y, X_scale: str = "minmax", y_scale: str=None):
    # scale

    # nan handling: onehot


def generate_data_scaler(method: str = "minmax"):
    """
    Scale the data using the specified method.
    """
    if method == "minmax":
        return MinMaxScaler()
    elif method == "standard":
        return StandardScaler()
    elif method == "log":
        return FunctionTransformer(log_transform)


def onehot_nan(df, discard_nanrows: bool = True):
    """
    One-hot encode the nan values in the data.

    TODO: should I just remove these rows outright?
    """
    nan_rows = df.isna().any(axis=1)

    if discard_nanrows:
        return df.dropna()
    else:
        # encode any rows containing nans to additional column
        df["nan_onehot"] = nan_rows.astype(int)
        # fill any nans with zeros
        return df.fillna(0)


def log_transform(x):
    return np.log(x + 1)


def sklearn_random_search(X, y, n_jobs: int = int(multiprocessing.cpu_count() * FRAC_COMPUTE)):
    """
    Perform a random search for the best hyperparameters for an XGBoost model.
    Must only fit on training set, to avoid leakage onto test/validation set.
    """
    print("Parallel Parameter optimization...")
    # Make sure the number of threads is balanced.
    xgb_model = xgb.XGBRegressor(
        n_jobs=n_jobs, tree_method="hist"
    )

    param_space = functions_creche.xgb_random_search(custom_scale_pos_weight=None)

    # time execution
    start_time = time.time()

    reg = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_space,
        cv=CV_FOLDS,
        n_iter=N_ITER,
        scoring="neg_root_mean_squared_error",  # TODO: look int other metrics
        verbose=1,
        n_jobs=n_jobs,
    )
    reg.fit(X, y)
    end_time = time.time()

    # Calculate and print the elapsed time
    elapsed_time = end_time - start_time
    print(f"Elapsed Time: {elapsed_time:.02f} seconds")

    print(f"Best score: {reg.best_score_:.02f}")
    print("Best parameters: ", reg.best_params_)


def xgb_baseline(dtrain):
    param_space = functions_creche.xgb_random_search(custom_scale_pos_weight=None)
    # randomly choose a value for each hyperparameter
    params = {key: np.random.choice(value) for key, value in param_space.items()}
    params["eval_metric"] = "rmse"
    # setting large num_boost_round, for which optimal number is hopefully less
    num_boost_round = 999

    # get mean value from training data
    mean_train = np.mean(y_train)
    # baseline predictions
    baseline_preds = np.full_like(y_train, mean_train)
    # baseline RMSE
    baseline_rmse = root_mean_squared_error(y_train, baseline_preds)



def xgb_random_search(dtrain: xgb.DMatrix, custom_scale_pos_weight=None):

    param_space = functions_creche.xgb_random_search(custom_scale_pos_weight=None)

    # time execution
    start_time = time.time()

    # for each combination of hyperparameters ()
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        seed=42,
        nfold=CV_FOLDS,
        metrics="rmse",
        early_stopping_rounds=EARLY_STOPPING_ROUNDS
    )
    
    
    reg.fit(X, y)
    end_time = time.time()

    # Calculate and print the elapsed time
    elapsed_time = end_time - start_time
    print(f"Elapsed Time: {elapsed_time} seconds")

    print(reg.best_score_)
    print(reg.best_params_)


# if __name__ == "__main__":
def main():
    # DATA LOADING
    # load data. TODO: add multifile loading for machines with reasonable numbers of CPUs
    print("loading static data...")
    all_data_df = load_static_data(RES_STR)

    # DATA PREPROCESSING
    # apply depth mask
    print("preprocessing...")
    depth_masked_df = functions_creche.depth_filter(all_data_df, depth_mask_lims=DEPTH_MASK_LIMS)
    # split data into train, test, validation
    [train_df, test_df, val_df] = functions_creche.train_test_val_split(
        depth_masked_df, ttv_fractions=TTV_FRACTIONS, split_method=SPLIT_METHOD)
    # preprocess data: initialise scalers values
    scaler_X = generate_data_scaler(SCALE_METHOD_X)
    scaler_y = generate_data_scaler(SCALE_METHOD_y)
    # preprocess data: nan handling
    [train_df, test_df, val_df] = [onehot_nan(df, REMOVE_NAN_ROWS) for df in [train_df, test_df, val_df]]
    # split into X and y for each subset
    (train_X, train_y), (test_X, test_y), (val_X, val_y) = [split_X_y(
        df, GROUND_TRUTH_NAME) for df in [train_df, test_df, val_df]]
    # fit scalers
    scaler_X.fit(train_X)
    scaler_y.fit(train_y)
    # transform data
    (train_X, train_y), (test_X, test_y), (val_X, val_y) = [
        (scaler_X.transform(X), scaler_y.transform(y)) for X, y in [
            (train_X, train_y), (test_X, test_y), (val_X, val_y)]]
    # cast data to DMatrix for native xgboost handling
    dtrain, dtest, dval = [xgb.DMatrix(X, y) for X, y in [
        (train_X, train_y), (test_X, test_y), (val_X, val_y)]]

    # MODEL FITTING
    print("fitting model...")
    sklearn_random_search(train_X[:DATA_SUBSET], train_y[:DATA_SUBSET])


if __name__ == "__main__":
    main()

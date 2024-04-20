# compute/file handling
import multiprocessing
from functools import lru_cache
import time
import random
from tqdm import tqdm
from pathlib import Path

# ml
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler, FunctionTransformer
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

# custom
from coralshift import functions_creche
from coralshift.dataloading import config

# general
import numpy as np
import pandas as pd
import xarray as xa

# COMPUTE SETUP
FRAC_COMPUTE = 0.25
CV_FOLDS = 3
N_ITER = 10
EARLY_STOPPING_ROUNDS = 10
EVAL_METRIC = "mae"
NUM_BOOST_ROUND = 100

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


def sklearn_random_search(
    X, y, n_jobs: int = int(multiprocessing.cpu_count() * FRAC_COMPUTE)
):
    """
    Perform a random search for the best hyperparameters for an XGBoost model.
    Must only fit on training set, to avoid leakage onto test/validation set.
    """
    print("Parallel Parameter optimization...")
    # Make sure the number of threads is balanced.
    xgb_model = xgb.XGBRegressor(n_jobs=n_jobs, tree_method="hist")

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


def generate_baseline_predictions(
    train_label: xgb.DMatrix | np.ndarray | pd.DataFrame,
    test_label: xgb.DMatrix | np.ndarray | pd.DataFrame,
):
    """
    Generate a baseline prediction (mean) for the given label.

    Args:
        label (xgb.DMatrix | np.ndarray | pd.DataFrame): The label to generate the baseline prediction for.

    Returns:
        np.ndarray: The baseline prediction.

    """
    # cast any dmatrices to np.ndarray
    if isinstance(train_label, xgb.DMatrix):
        train_label = train_label.get_label()
    if isinstance(test_label, xgb.DMatrix):
        test_label = test_label.get_label()

    return np.full_like(test_label, np.mean(train_label))


def calculate_score(
    eval_metric: str,
    label: xgb.DMatrix | np.ndarray | pd.DataFrame,
    predictions: np.ndarray | pd.DataFrame,
):
    """
    Calculate the evaluation metric specified.

    Args:
        eval_metric (str): The evaluation metric to use.
        label (xgb.DMatrix | np.ndarray | pd.DataFrame): The label to calculate the metric for.
        predictions (np.ndarray | pd.DataFrame): The predictions to calculate the metric for.

    Returns:
        float: The calculated score.
    """
    if isinstance(label, xgb.DMatrix):
        label = label.get_label()

    if eval_metric == "rmse":
        return root_mean_squared_error(label, predictions)
    elif eval_metric == "mae":
        return mean_absolute_error(label, predictions)
    else:
        print(f"To implement: {eval_metric}")


def xgb_baseline(dtrain, dtest, eval_metric: str = "rmse"):
    """Baseline model performance, determination of number of boosting rounds

    Args:
        dtrain (xgb.DMatrix): The training data.
        dtest (xgb.DMatrix): The test data.
        eval_metric (str, optional): The evaluation metric to use. Defaults to "rmse".
    """
    param_space = functions_creche.xgb_random_search(custom_scale_pos_weight=None)
    # randomly choose a value for each hyperparameter
    params = {key: np.random.choice(value) for key, value in param_space.items()}
    params["eval_metric"] = eval_metric
    # setting large num_boost_round, for which optimal number is hopefully less
    baseline_model = xgb.train(
        params,
        dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        evals=[(dtest, "test")],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )

    print(
        "Best baseline {}: {:.2f} with {}/{} rounds".format(
            eval_metric,
            baseline_model.best_score,
            baseline_model.best_iteration + 1,
            NUM_BOOST_ROUND,
        )
    )

    return baseline_model


def random_sample_param_space(param_space: dict, num_samples: int = 5):
    """
    Randomly sample a dictionary of parameter values to create "num_samples" combinations of values.

    Args:
        param_space (dict): Dictionary of parameter names and their possible values.
        num_samples (int): Number of combinations to sample.

    Returns:
        list: List of dictionaries containing the sampled combinations of parameter values.
    """
    param_keys = list(param_space.keys())
    sampled_combinations_set = set()

    # Loop to randomly sample values
    for _ in range(num_samples):
        # Randomly sample values for each key
        sampled_values = {key: random.choice(param_space[key]) for key in param_keys}

        # Check if the combination has already been sampled
        while tuple(sampled_values.items()) in sampled_combinations_set:
            sampled_values = {
                key: random.choice(param_space[key]) for key in param_keys
            }

        # Add the sampled combination to the set
        sampled_combinations_set.add(tuple(sampled_values.items()))

    return [dict(combination) for combination in sampled_combinations_set]


def xgb_random_search(
    dtrain: xgb.DMatrix, num_boost_round: int = 999, custom_scale_pos_weight=None
):

    param_space = functions_creche.xgb_random_search(custom_scale_pos_weight=None)

    param_sets = random_sample_param_space(param_space, num_samples=N_ITER)

    # time execution
    start_time = time.time()

    min_mae = float("Inf")

    # for each combination of hyperparameters in param_space, cross-fold evaluate
    for param_set in tqdm(param_sets):
        cv_results = xgb.cv(
            param_set,
            dtrain,
            num_boost_round=num_boost_round,  # TODO: global?
            seed=42,
            nfold=CV_FOLDS,
            metrics={EVAL_METRIC},
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        )

        mean_mae = cv_results["test-mae-mean"].min()
        boost_rounds = cv_results["test-mae-mean"].argmin()
        print(f"\tMAE {mean_mae:.05f} for {boost_rounds+1}/{NUM_BOOST_ROUND} rounds")
        if mean_mae < min_mae:
            min_mae = mean_mae

    end_time = time.time()

    # Calculate and print the elapsed time
    elapsed_time = end_time - start_time
    print(f"Elapsed Time: {elapsed_time} seconds")

    return cv_results
    best_params = cv_results.best_params_
    print(f"Best params: {best_params}, MAE: {min_mae}")


def main():
    # DATA LOADING AND PREPROCESSING
    (train_X, train_y), (test_X, test_y), (val_X, val_y) = functions_creche.load_data(
        res_str=RES_STR,
        depth_mask_lims=DEPTH_MASK_LIMS,
        ground_truth_name=GROUND_TRUTH_NAME,
        static_data_dir_fp=STATIC_DATA_DIR_FP,
        ttv_fractions=TTV_FRACTIONS,
        split_method=SPLIT_METHOD,
        scale_method_X=SCALE_METHOD_X,
        scale_method_y=SCALE_METHOD_y,
        remove_nan_rows=REMOVE_NAN_ROWS,
    )
    # cast data to DMatrix for native xgboost handling
    dtrain, dtest, dval = [
        xgb.DMatrix(X, y)
        for X, y in [(train_X, train_y), (test_X, test_y), (val_X, val_y)]
    ]

    # BASELINE MODEL
    baseline_preds = generate_baseline_predictions(dtrain, dtest)
    baseline_score = calculate_score(EVAL_METRIC, dtest, baseline_preds)
    print(f"Baseline {EVAL_METRIC}: {baseline_score}")
    # calculate number of boosting rounds
    baseline_model = xgb_baseline(dtrain, dtest, eval_metric=EVAL_METRIC)

    # LOGREG MODEL FITTING
    print("fitting log-reg (maxent) model...")

    # XGBOOST MODEL FITTING
    # print("fitting model...")
    # sklearn_random_search(train_X[:DATA_SUBSET], train_y[:DATA_SUBSET])
    # xgb_random_search(dtrain, num_boost_round=NUM_BOOST_ROUND)

    return baseline_model, train_X, train_y


if __name__ == "__main__":
    main()

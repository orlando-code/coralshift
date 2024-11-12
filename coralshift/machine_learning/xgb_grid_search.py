# file handling
import logging
import argparse
from pathlib import Path
import re

# general
import itertools
from datetime import datetime
import pandas as pd

# machine learning
import xgboost as xgb
import os

# custom
from coralshift.machine_learning import static_models


def parse_hyperparameter_search_log_file(file_path: Path):
    """
    Parses a log file to extract already completed parameter searches.
    Returns a set of parameter tuples.
    """
    completed_searches = set()
    pattern = re.compile(r"Testing parameters: (.+)")

    with open(file_path, "r") as file:
        for line in file:
            match = pattern.search(line)
            if match:
                # Parse the parameters string and convert to a tuple
                params_str = match.group(1)
                params_dict = eval(params_str)
                params_tuple = tuple(params_dict.values())
                completed_searches.add(params_tuple)
    return completed_searches


def parse_hyperparameter_logs_from_directory(directory_path: Path):
    """
    Parses all log files in the specified directory to get completed searches.
    """
    log_file_pattern = re.compile(r".*xgb_grid_search_.*\.log$")
    all_completed = set()

    directory_path = Path(directory_path)
    if not directory_path.is_dir():
        raise ValueError(f"Provided path is not a directory: {directory_path}")

    for file_path in directory_path.iterdir():
        if file_path.is_file() and log_file_pattern.match(file_path.name):
            parsed_results = parse_hyperparameter_search_log_file(file_path)
            all_completed.update(parsed_results)
        elif file_path.is_file():
            raise ValueError(f"File does not match expected pattern: {file_path}")

    return all_completed


def get_remaining_search_combinations(all_combinations, completed_searches):
    """
    Filters out combinations that have already been completed.
    """
    return [combo for combo in all_combinations if combo not in completed_searches]


def main(data_fp, label_str, num_boost_round, num_search_vals, cv, seed):
    # Set up logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging_dir = Path("logs/xgb_grid_search")
    logging_dir.mkdir(parents=True, exist_ok=True)
    logging_fp = logging_dir / f"xgb_grid_search_{timestamp}.log"
    logging.basicConfig(
        filename=logging_fp,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    logging.info("Starting grid search process.")

    # Load your dataset
    all_df = pd.read_parquet(data_fp)
    all_X = all_df.drop(columns=[label_str])
    all_y = all_df[label_str]
    print("\nNumber of features", len(all_X.columns), flush=True)
    print("Number of samples", len(all_y), flush=True)
    dtrain = xgb.DMatrix(data=all_X, label=all_y)

    # Generate search grid
    params_grid = static_models.xgb_search_grid(n_trials=num_search_vals)

    # Convert params grid dictionary to a list of parameter combinations
    param_combinations = list(itertools.product(*params_grid.values()))
    param_keys = list(params_grid.keys())

    # Check for previously completed searches
    completed_searches = parse_hyperparameter_logs_from_directory(logging_dir)
    remaining_combinations = get_remaining_search_combinations(
        param_combinations, completed_searches
    )

    logging.info(f"Total parameter combinations: {len(param_combinations)}")
    logging.info(f"Completed searches: {len(completed_searches)}")
    logging.info(f"Remaining searches: {len(remaining_combinations)}")

    # Perform grid search over remaining combinations
    results = []
    for param_values in remaining_combinations:
        # Create a dictionary for the current parameter combination
        params = dict(zip(param_keys, param_values))
        params["eval_metric"] = "rmse"
        logging.info(f"Testing parameters: {params}")

        # Run cross-validation
        try:
            cv_results = xgb.cv(
                params=params,
                dtrain=dtrain,
                num_boost_round=num_boost_round,
                nfold=cv,
                seed=seed,
                callbacks=[xgb.callback.EarlyStopping(10)],
                verbose_eval=False,
            )

            # Get the mean RMSE and standard deviation across folds for the best iteration
            best_iteration = cv_results["test-rmse-mean"].idxmin()
            best_rmse_mean = cv_results.loc[best_iteration, "test-rmse-mean"]
            best_rmse_std = cv_results.loc[best_iteration, "test-rmse-std"]

            # Log the results for this combination
            logging.info(
                f"Best iteration: {best_iteration}, RMSE: {best_rmse_mean}, RMSE std: {best_rmse_std}"
            )

            # Store results
            results.append(
                {
                    **params,
                    "best_iteration": best_iteration,
                    "test_rmse_mean": best_rmse_mean,
                    "test_rmse_std": best_rmse_std,
                }
            )

        except Exception as e:
            logging.error(f"Error with parameters {params}: {e}")
            continue

    # Save results to a CSV file
    results_df = pd.DataFrame(results)
    results_fp = logging_dir / f"xgb_results_{timestamp}.csv"
    results_df.to_csv(results_fp, index=False)

    # Log the best parameters and their metrics
    if not results_df.empty:
        best_params = results_df.loc[results_df["test_rmse_mean"].idxmin()]
        logging.info("Best parameters found:")
        logging.info(best_params)
        print("Grid search completed. Check the log and CSV file for details.")
    else:
        print("No results to save.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform grid search for XGBoost parameters."
    )
    parser.add_argument(
        "--data_fp",
        default="/maps/rt582/coralshift/coralshift/machine_learning/xgb_grid_search.py",
        type=str,
        required=True,
        help="File path to the dataset (parquet required).",
    )
    parser.add_argument(
        "--num_boost_round", type=int, default=1000, help="Number of boosting rounds."
    )
    parser.add_argument(
        "--label_str",
        type=str,
        default="UNEP_GDCR",
        help="Label string for the target variable.",
    )
    parser.add_argument(
        "--num_search_vals",
        type=int,
        default=4,
        help="Number of values to search in each parameter.",
    )
    parser.add_argument(
        "--cv", type=int, default=5, help="Number of cross-validation folds."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--num_cores", type=int, default=160, help="Number of cores to use."
    )
    args = parser.parse_args()

    os.environ["OMP_NUM_THREADS"] = str(args.num_cores)

    main(
        data_fp=args.data_fp,
        label_str=args.label_str,
        num_boost_round=args.num_boost_round,
        num_search_vals=args.num_search_vals,
        cv=args.cv,
        seed=args.seed,
    )

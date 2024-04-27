# general
import numpy as np
import pandas as pd

# machine learning
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
)
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

# distribution
import joblib

# file handling
from pathlib import Path
import time

# custom
from coralshift.processing import ml_processing
from coralshift.plotting import model_results
from coralshift.utils import file_ops, utils, config


def run_config_files(
    fps: list[str],
    ds_info: dict = None,
    trains: tuple[pd.DataFrame, pd.DataFrame] = None,
    tests: tuple[pd.DataFrame, pd.DataFrame] = None,
    vals: tuple[pd.DataFrame, pd.DataFrame] = None,
):
    # for fp in tqdm(
    #     fps, total=len(list(fps)), desc="Running models according to config file(s)"
    # ):
    for fp in fps:
        config_info = file_ops.read_yaml(fp)

        if not (ds_info and trains and tests and vals):
            # get data
            (trains, tests, vals), ds_info = ml_processing.ProcessMLData(
                config_info=config_info
            ).generate_ml_ready_data()
        # train model
        model = RunStaticML(
            trains=trains,
            tests=tests,
            vals=vals,
            config_info=config_info,
            additional_info=ds_info,
        ).run_model()

        fp_save_dir = RunStaticML(
            config_info=config_info, additional_info=ds_info
        ).construct_fp_dir()

        latest_config_fp_stem = (
            fp_save_dir
            / RunStaticML(
                config_info=config_info, additional_info=ds_info
            ).generate_latest_unique_fp()
        )
        new_config_fp = f"{latest_config_fp_stem}.yaml"
        # get updated config file
        # analyse results
        model_results.AnalyseResults(
            model=model,
            trains=trains,
            tests=tests,
            vals=vals,
            config_info=file_ops.read_yaml(new_config_fp),
        ).analyse_results()


class RunStaticML:
    def __init__(
        self,
        trains: tuple = None,
        tests: tuple = None,
        vals: tuple = None,
        config_info: dict = None,
        additional_info: dict = None,  # hacky, but used to get summary of ds into new config file
    ):
        self.trains = trains
        self.tests = tests
        self.vals = vals
        self.regressor_classification_threshold = config_info[
            "regressor_classification_threshold"
        ]
        self.n_samples = config_info["hyperparameter_search"]["n_samples"]
        self.model_code = config_info["model_code"]
        self.cv_folds = config_info["hyperparameter_search"]["cv_folds"]
        self.n_iter = config_info["hyperparameter_search"]["n_iter"]
        self.do_search = config_info["hyperparameter_search"]["do_search"]
        self.n_trials = config_info["hyperparameter_search"]["n_trials"]
        self.search_type = config_info["hyperparameter_search"]["type"]
        self.do_train = config_info["do_train"]
        self.do_save_model = config_info["do_save_model"]
        self.config_info = config_info
        self.additional_info = additional_info
        if config_info:
            self.__dict__.update(config_info)

    def initialise_model(self, params: dict = None):
        # get instance of any model type
        return ModelInitializer(model_type=self.model_code, params=params).get_model()

    def threshold_datasets(self, dss: list[tuple[pd.DataFrame, pd.Series]]):
        # return list of tuples of thresholded datasets
        for ds in dss:
            yield self.threshold_dataset(ds)

    def threshold_dataset(
        self,
        ds: tuple[pd.DataFrame, pd.Series],
        # trains=None, tests=None, vals=None
    ):
        if ModelInitializer(model_type=self.model_code).get_data_type() == "discrete":
            return ds[0], ml_processing.cont_to_class(ds[1])
        else:
            return ds

    def xgboost_formatting(self, trains=None, tests=None, vals=None):
        if "xgb" in self.model_code:
            # convert data to DMatrix format
            dtrains = xgb.DMatrix(trains[0], trains[1])
            dtests = xgb.DMatrix(tests[0], tests[1])
            dvals = xgb.DMatrix(vals[0], vals[1])
            return dtrains, dtests, dvals
        else:
            return trains, tests, vals

    def get_param_search_grid(self, params: dict = None):
        # if no params provided, use default values
        if not params:
            # if param_search is True, run grid search or random search dependent on input
            if self.config_info["hyperparameter_search"]["type"] == "random":
                return ModelInitializer(
                    model_type=self.model_code
                ).get_random_search_grid()
            elif (
                self.config_info["hyperparameter_search"]["type"] == "grid"
            ):  # TODO: allow chaining of random and grid searches (including saving optimal values)
                return ModelInitializer(
                    model_type=self.model_code
                ).get_grid_search_grid()
            else:
                raise ValueError(
                    f"Parameter search type {self.config_info['hyperparameter_search']['type']} not recognised."
                )
        else:
            print("ASdfasd")

    def get_param_search(self, search_type: str, search_grid: dict = None):
        model = self.initialise_model()
        # if search grid not specified (i.e. random search?)
        if not search_grid:
            search_grid = self.get_param_search_grid()

        if search_type == "random":
            return RandomizedSearchCV(
                model,
                search_grid,
                cv=self.cv_folds,
                n_iter=self.n_iter,
                verbose=1,
                # TODO: replace jobs hardcoding AH .g. n_jobs: int = int(multiprocessing.cpu_count() * FRAC_COMPUTE)?
                n_jobs=64,
            )
        elif search_type == "grid":
            search_grid = generate_gridsearch_parameter_grid(
                search_grid
            )  # currently not able to specify number of values for grid search
            return GridSearchCV(model, search_grid, cv=self.cv_folds, verbose=1)
        else:
            raise ValueError(
                f"Parameter search type {self.config_info['hyperparameter_search']['type']} not recognised."
            )

    def do_parallel_search(self, search_object):
        print(f"\nRunning parameter search for {self.model_code}...")
        with joblib.parallel_config(backend="dask", verbose=1):
            search_object.fit(
                self.trains[0][: self.n_samples],
                self.trains[1][: self.n_samples],
            )
        return search_object.best_params_

    def fetch_best_params_from_config(self):
        best_params = None
        # Try to fetch best_grid_params first, then best_random_params
        try:
            params_fp = self.config_info["file_paths"]["best_grid_params"]
            print(f"Loading best parameters from {params_fp}...")
            best_params = file_ops.read_pkl(params_fp)
        except KeyError:
            try:
                params_fp = self.config_info["file_paths"]["best_random_params"]
                print(f"Loading best parameters from {params_fp}...")
                best_params = file_ops.read_pkl(params_fp)
            except KeyError:
                print(
                    "No best parameters file(s) found in CONFIG file. Using default values instead."
                )
        return best_params

    def save_param_search(
        self, fp_root: str | Path, best_params: dict, search_type: str
    ):
        if search_type == "grid":
            params_fp = f"{fp_root}_PARAM_GRID.pickle"
        elif search_type == "random":
            params_fp = f"{fp_root}_PARAM_RANDOM.pickle"
        if not best_params:  # if best_params are None (not specified)
            params_fp = f"{fp_root}_PARAM_GENERIC.pickle"

        print(f"\nSaving best parameters to {params_fp}...")
        file_ops.write_pkl(params_fp, best_params)
        return params_fp

    def do_param_searches(self, fp_root: Path | str):
        if self.do_search:
            search_types = self.config_info["hyperparameter_search"]["search_types"]
            # if random and grid
            if len(search_types) > 2:
                print(f"Unexpected number of search types: {search_types}.")
            elif all(substring in search_types for substring in ["grid", "random"]):
                # do random
                search_object = self.get_param_search(search_type="random")
                print("RANDOM SEARCH")
                best_params = self.do_parallel_search(search_object)
                # save parameters
                self.save_param_search(fp_root, best_params, search_type="random")
                # do grid
                search_object = self.get_param_search(
                    search_type="grid", search_grid=best_params
                )
                print("GRID SEARCH")
                best_params = self.do_parallel_search(search_object)
                # save parameters
                self.save_param_search(fp_root, best_params, search_type="grid")
            # if can find random parameter file, but not grid, and grid specified run grid on random
            elif (
                Path(f"{fp_root}_PARAM_RANDOM.pickle").exists()
                and "grid" in search_types
            ):
                search_object = self.get_param_search(search_type="grid")
                best_random_params = self.fetch_best_params_from_config()
                best_params = self.do_parallel_search(
                    search_object, search_type="grid", search_grid=best_random_params
                )
                # save parameters
                self.save_param_search(fp_root, best_params, search_type="grid")
            # if only one specified, do it? ####
            else:
                search_object = self.get_param_search(search_type=search_types[0])
                best_params = self.do_parallel_search(search_object)
        else:
            # returns best of best params (grid first, then random, else None)
            best_params = self.fetch_best_params_from_config()

        return best_params

    def construct_fp_dir(self):
        res_str = utils.replace_dot_with_dash(str(self.config_info["resolution"]))
        fp_dir = Path(f"{config.runs_dir}/{res_str}d/{self.model_code}/")
        if not fp_dir.exists():
            fp_dir.mkdir(parents=True, exist_ok=True)
        return fp_dir

    def construct_fp_stem(self, fp_dir: Path | str, suffix: str = "pickle"):
        return "_".join(self.config_info["datasets"])

    def unique_identifier(self, fp_dir: Path | str, fp_stem: str):
        counter = 0
        new_filename = f"ID000_{fp_stem}_CONFIG"
        while list(fp_dir.glob(f"{new_filename}.*")) != []:
            counter += 1
            new_filename = f"ID{utils.pad_number_with_zeros(counter, resulting_len=3)}_{Path(fp_stem).stem}_CONFIG"
        return f"ID{utils.pad_number_with_zeros(counter, resulting_len=3)}"

    def get_highest_unique_filename(self, fp_dir: Path | str, fp_stem: str):
        # TODO: return generic fp, rather than CONFIG specifically?
        """Fetch the highest unique ID for a particular run"""
        counter = 0
        new_filename = f"ID000_{fp_stem}_CONFIG"
        while list(fp_dir.glob(f"{new_filename}.*")) != []:
            previous_fname = new_filename
            counter += 1
            new_filename = f"ID{utils.pad_number_with_zeros(counter, resulting_len=3)}_{Path(fp_stem).stem}_CONFIG"
        return previous_fname

    def train_model(self, fp_root: Path | str, hyperparams: dict = None):
        # initialise model with provided hyperparameters
        model = self.initialise_model(params=hyperparams)
        # train model
        if self.do_train:
            print(
                f"\n\nTraining the {self.model_code} model on {self.n_samples} datapoints..."
            )
            # sklearn models
            if "xgb" not in self.model_code:
                model.fit(self.trains[0], self.trains[1])
            # xgboost models
            elif self.config_info["ds_type"] == "static":
                trains, _, _ = self.xgboost_formatting(
                    self.trains, self.tests, self.vals
                )
                model = xgb.train(hyperparams, trains)
            else:
                print("timeseries models yet to be implemented")

        if self.config_info["do_save_model"]:
            model_fp = f"{fp_root}_MODEL.pickle"
            print(f"Saving model to {model_fp}...")
            file_ops.write_pkl(model_fp, model)

        return model

    def generate_latest_unique_fp(self):
        # generate path for saving data
        save_dir = self.construct_fp_dir()
        save_fp_stem = self.construct_fp_stem(save_dir)
        return self.get_highest_unique_filename(save_dir, save_fp_stem)

    def generate_fp_root(self):
        # generate path for saving data
        save_dir = self.construct_fp_dir()
        save_fp_stem = self.construct_fp_stem(save_dir)
        # generate unique ID code dependent on which files already exist
        unique_identifier = self.unique_identifier(
            fp_dir=save_dir, fp_stem=save_fp_stem
        )
        return save_dir / f"{unique_identifier}_{save_fp_stem}"

    def run_model(self):
        # threshold labels if necessary
        self.trains, self.tests, self.vals = self.threshold_datasets(
            [self.trains, self.tests, self.vals]
        )
        fp_root = self.generate_fp_root()

        search_start_time = time.time()
        best_params = self.do_param_searches(fp_root)
        search_time = time.time() - search_start_time
        params_fp = f"{fp_root}_PARAMS.pickle"  # TODO: adjust this to return random or grid search in name
        file_ops.write_pkl(params_fp, best_params)

        train_start_time = time.time()
        model = self.train_model(fp_root=fp_root, hyperparams=best_params)
        train_time = time.time() - train_start_time
        # save config file
        config_fp = f"{fp_root}_CONFIG.yaml"
        file_ops.save_yaml(config_fp, self.config_info)

        # add file information to yaml
        file_info = {
            "file_paths": {
                "model": f"{fp_root}_MODEL.pickle" if self.do_save_model else None,
                "best_params": f"{fp_root}_PARAMS.pickle" if self.do_search else None,
                "config": f"{fp_root}_CONFIG.yaml",
                "search_time": f"{search_time:.02f}s",
                "train_time": f"{train_time:.02f}s",
            },
            "additional_info": self.additional_info if self.additional_info else None,
        }
        file_ops.edit_yaml(config_fp, file_info)
        return model


class ModelInitializer:
    def __init__(self, model_type: str, random_state: int = 42, params: dict = None):
        self.random_state = random_state
        self.model_info = [
            # discrete models
            {
                "model_type": "logreg",
                "model_type_full_name": "Logistic Regression",
                "data_type": "discrete",
                "model": LogisticRegression(
                    class_weight="balanced",
                    n_jobs=-1,
                    verbose=1,
                    random_state=self.random_state,
                ),
                "search_grid": logreg_search_grid(),
            },
            {
                "model_type": "rf_cf",
                "model_type_full_name": "Random Forest Classifier",
                "data_type": "discrete",
                "model": RandomForestClassifier(
                    class_weight="balanced",
                    n_jobs=-1,
                    verbose=1,
                    random_state=self.random_state,
                ),
                "search_grid": rf_search_grid(),
            },
            {
                "model_type": "gb_cf",
                "model_type_full_name": "Gradient Boosting Classifier",
                "data_type": "discrete",
                "model": GradientBoostingClassifier(
                    verbose=1,
                    random_state=self.random_state,  # N.B. has sample weight rather than class weight. TODO: implement
                ),
                "search_grid": boosted_search_grid(),
            },
            {
                "model_type": "rf_reg",
                "model_type_full_name": "Random Forest Regressor",
                "data_type": "continuous",
                "model": RandomForestRegressor(
                    verbose=1, n_jobs=-1, random_state=self.random_state
                ),
                "search_grid": rf_search_grid(),
            },
            {
                "model_type": "gb_reg",
                "model_type_full_name": "Gradient Boosting Regressor",
                "data_type": "continuous",
                "model": GradientBoostingRegressor(
                    verbose=1, random_state=self.random_state
                ),
                "search_grid": boosted_search_grid(model_type="regressor"),
            },
            {
                "model_type": "xgb_cf",
                "model_type_full_name": "XGBoost Classifier",
                "data_type": "discrete",
                "model": xgb.XGBClassifier(verbose=1, random_state=self.random_state),
                "search_grid": xgb_search_grid(),
            },
            {
                "model_type": "xgb_reg",
                "model_type_full_name": "XGBoost Regressor",
                "data_type": "continuous",
                "model": xgb.XGBRegressor(verbose=1, random_state=self.random_state),
                "search_grid": xgb_search_grid(model_type="regressor"),
            },
            {
                "model_type": "mlp_cf",
                "model_type_full_name": "MLP Classifier",
                "data_type": "discrete",
                "model": MLPClassifier(
                    verbose=1, random_state=self.random_state, max_fun=15000
                ),
                "search_grid": mlp_search_grid(),
            },
            {
                "model_type": "mlp_reg",
                "model_type_full_name": "MLP Regressor",
                "data_type": "continuous",
                "model": MLPRegressor(
                    verbose=1, random_state=self.random_state, max_fun=15000
                ),
                "search_grid": mlp_search_grid(),
            },
        ]
        self.model_type = self._normalize_model_name(model_type)
        self.params = params

    def _normalize_model_name(self, model_name):
        normalized_names = {
            info["model_type_full_name"]: info["model_type"] for info in self.model_info
        }
        return normalized_names.get(model_name, model_name)

    def get_data_type(self):
        for m in self.model_info:
            if m["model_type"] == self.model_type:
                return m["data_type"]
        else:
            raise ValueError(f"'{self.model_type}' not a valid model.")

    def get_model(self):
        for m in self.model_info:
            if m["model_type"] == self.model_type:
                model_instance = m["model"]
                if self.params:
                    model_instance.set_params(**self.params)
                return model_instance
        else:
            raise ValueError(f"'{self.model_type}' not a valid model.")

    def get_random_search_grid(self):
        for m in self.model_info:
            if m["model_type"] == self.model_type:
                return m["search_grid"]
        else:
            raise ValueError(f"'{self.model_type}' not a valid model.")


#
def make_vals_list(val_lims: tuple, num_vals: int, spacing: str = "linear") -> list:
    if spacing == "linear":
        return equal_spacing(val_lims, num_vals)
    elif spacing == "log":
        return log_spacing(val_lims, num_vals)
    else:
        raise ValueError("Spacing must be either 'linear' or 'log'.")


def equal_spacing(val_lims: tuple, num_vals: int) -> list:
    if all(isinstance(x, int) for x in val_lims):
        return np.linspace(
            start=val_lims[0], stop=val_lims[1], num=num_vals, dtype=int
        ).tolist()
    elif any(isinstance(x, float) for x in val_lims):
        return np.linspace(start=val_lims[0], stop=val_lims[1], num=num_vals).tolist()
    else:
        raise ValueError("Values in val_lims must be either floats or integers.")


def log_spacing(val_lims: tuple, num_vals: int) -> list:
    if all(isinstance(x, int) for x in val_lims):
        return [
            int(val) for val in np.logspace(*np.log10(val_lims), num=num_vals).tolist()
        ]
    else:
        return np.logspace(*np.log10(val_lims), num=num_vals).tolist()


def rf_search_grid(
    estimator_lims: tuple[int] = (200, 2000),
    max_features: list[str] = ["auto", "sqrt"],
    max_depth_lims: tuple[int] = (10, 110),
    min_samples_split: list[int] = [2, 5, 10],
    min_samples_leaf: list[int] = [1, 2, 4],
    bootstrap: list[bool] = [True, False],
) -> dict:
    # Number of trees in random forest
    n_estimators = [
        int(x)
        for x in np.linspace(
            start=min(estimator_lims), stop=max(estimator_lims), num=10
        )
    ]
    # Number of features to consider at every split
    max_features = max_features
    # Maximum number of levels in tree
    max_depth = [
        int(x) for x in np.linspace(min(max_depth_lims), max(max_depth_lims), num=11)
    ]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = min_samples_split
    # Minimum number of samples required at each leaf node
    min_samples_leaf = min_samples_leaf
    # Method of selecting samples for training each tree
    bootstrap = bootstrap
    # Create the random grid
    random_grid = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "bootstrap": bootstrap,
    }
    return random_grid


def mlp_search_grid(
    hidden_layer_sizes=[(500,), (100,), (50, 30, 20)],
    activation: list[str] = ["identity", "logistic", "tanh", "relu"],
    solver: list[str] = ["adam", "sgd", "lbfgs"],
    alpha_lims: tuple[float] = (0.000001, 0.001),
    batch_size=["auto"],
    learning_rate=["adaptive"],
    learning_rate_init_lims: tuple[float] = (0.0001, 0.01),
    max_iter_lims: tuple[int] = (50, 500),
    shuffle: tuple[bool] = [True, False],
    momentum_lims: tuple[float] = (0.8, 1),
    nesterovs_momentum: tuple[bool] = [True, False],
    # max_fun=15000,
    n_trials=3,
) -> dict:
    # TODO: change hidden layer sizes
    alpha = make_vals_list(alpha_lims, n_trials, "log")
    learning_rate_init = make_vals_list(learning_rate_init_lims, n_trials)
    max_iter = make_vals_list(max_iter_lims, n_trials)
    momentum = make_vals_list(momentum_lims, n_trials)

    return {
        "hidden_layer_sizes": hidden_layer_sizes,
        "activation": activation,
        "solver": solver,
        "alpha": alpha,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "learning_rate_init": learning_rate_init,
        "max_iter": max_iter,
        "shuffle": shuffle,
        "momentum": momentum,
        "nesterovs_momentum": nesterovs_momentum,
        # "max_fun": max_fun,
    }


def boosted_search_grid(
    loss: list[str] = ["squared_error", "absolute_error", "huber", "quantile"],
    learning_rate_lims: tuple[float] = (0.001, 1.0),
    n_estimators_lims: tuple[int] = (100, 2000),
    subsample_lims: tuple[float] = (0.1, 1.0),
    criterion: list[str] = ["friedman_mse", "squared_error"],
    min_samples_split_lims: tuple[int] = (2, 20),
    min_samples_leaf_lims: tuple[float] = (0.0, 1.0),
    min_weight_fraction_leaf_lims: tuple[float] = (0, 0.5),
    max_depth_lims: tuple[int] = (1, 10),
    min_impurity_decrease: tuple[float] = (0.001, 10000),
    max_features: list[str] = ["sqrt", "log2", None],
    max_leaf_nodes_lims: list[int] = [2, 1000],
    ccp_alpha_lims: tuple[float] = (0.001, 10000),
    model_type: str = "classifier",
    n_trials: int = 3,
) -> dict:
    # Learning rate (shrinkage)
    learning_rate = make_vals_list(learning_rate_lims, n_trials, "log")
    # Number of trees in the ensemble
    n_estimators = make_vals_list(n_estimators_lims, n_trials)
    # Fraction of samples to be used for training each tree
    subsample = make_vals_list(subsample_lims, n_trials)
    min_samples_split = make_vals_list(min_samples_split_lims, n_trials)
    min_samples_leaf = make_vals_list(min_samples_leaf_lims, n_trials)
    min_weight_fraction_leaf = make_vals_list(min_weight_fraction_leaf_lims, n_trials)
    max_depth = make_vals_list(max_depth_lims, n_trials, "log")
    min_impurity_decrease = make_vals_list(min_impurity_decrease, n_trials, "log")
    max_leaf_nodes = make_vals_list(max_leaf_nodes_lims, n_trials, "log")
    ccp_alpha_lims = make_vals_list(ccp_alpha_lims, n_trials, "log")

    # Loss function to optimize
    if model_type == "classifier":
        loss = ["exponential", "log_loss"]

    # Create the random grid
    return {
        "loss": loss,
        "learning_rate": learning_rate,
        "n_estimators": n_estimators,
        "subsample": subsample,
        "criterion": criterion,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "min_weight_fraction_leaf": min_weight_fraction_leaf,
        "max_depth": max_depth,
        "min_impurity_decrease": min_impurity_decrease,
        "max_leaf_nodes": max_leaf_nodes,
        "max_features": max_features,
    }


def xgb_search_grid(
    n_estimators_lims: tuple[int] = (100, 2000),
    learning_rate_lims: tuple[float] = (0.001, 1.0),
    max_depth_lims: tuple[int] = (1, 10),
    min_samples_split: list[int] = [2, 5, 10],
    min_samples_leaf: list[int] = [1, 2, 4],
    max_features: list[str] = ["auto", "sqrt"],
    loss: list[str] = ["ls", "lad", "huber", "quantile"],
    subsample_lims: tuple[float] = (0.1, 1.0),
    criterion: list[str] = ["friedman_mse", "mse"],
    model_type: str = "classifier",
) -> dict:
    # TODO: there are more parameters here, some of which may depend on the booster and so throw a load of errors
    # look in graveyard at xgb_random_search
    #
    # Number of trees in the ensemble
    n_estimators = [
        int(x)
        for x in np.linspace(
            start=min(n_estimators_lims), stop=max(n_estimators_lims), num=10
        )
    ]
    # Learning rate (shrinkage)
    learning_rate = np.logspace(*np.log10(learning_rate_lims), num=10).tolist()
    # Maximum depth of each tree
    max_depth = [
        int(x)
        for x in np.linspace(
            start=min(max_depth_lims), stop=max(max_depth_lims), num=10
        )
    ]
    max_depth.append(None)  # what is this doing?
    # Minimum number of samples required to split a node
    min_samples_split = min_samples_split
    # Minimum number of samples required at each leaf node
    min_samples_leaf = min_samples_leaf
    # Maximum number of features to consider at each split
    max_features = max_features
    # Loss function to optimize
    if model_type == "classifier":
        loss = ["exponential", "log_loss"]
        criterion = ["friedman_mse", "squared_error"]
    # Fraction of samples to be used for training each tree
    subsample = np.linspace(
        start=subsample_lims[0], stop=subsample_lims[1], num=10
    ).tolist()

    # Create the random grid
    random_grid = {
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "max_features": max_features,
        "loss": loss,
        "subsample": subsample,
        "criterion": criterion,
    }
    return random_grid


def logreg_search_grid(
    penalty: list[str] = [
        # "l1", "elasticnet"
        "l2"
    ],
    dual: list[bool] = [
        # True,
        False  # clashes with solver
    ],
    tol: list[float] = [1e-4, 1e-3, 1e-2],
    C: list[float] = [0.1, 1.0, 10.0],
    fit_intercept: list[bool] = [True, False],
    intercept_scaling: list[float] = [1.0, 2.0, 5.0],
    solver: list[str] = [
        "sag",
        "saga",
        # , "newton-cholesky" # some clashes with penalties
    ],
    max_iter: list[int] = [100, 200, 500],
    multi_class: list[str] = ["auto", "ovr", "multinomial"],
    verbose: list[int] = [0, 1, 2],
    warm_start: list[bool] = [True, False],
) -> dict:

    # Create the grid
    grid = {
        "penalty": penalty,
        "dual": dual,
        "tol": tol,
        "C": C,
        "fit_intercept": fit_intercept,
        "intercept_scaling": intercept_scaling,
        "solver": solver,
        "max_iter": max_iter,
        "multi_class": multi_class,
        "verbose": verbose,
        "warm_start": warm_start,
    }

    return grid


def generate_gridsearch_parameter_grid(params_dict: dict, num_samples: int = 2) -> dict:
    grid_params = {}
    for key, value in params_dict.items():
        if key == "verbose":
            grid_params[key] = [value]
        elif isinstance(value, bool) or isinstance(value, str):
            grid_params[key] = [value]
        elif isinstance(value, int) or isinstance(value, float):
            if num_samples == 0:
                raise ValueError("num_samples cannot be zero")
            elif num_samples % 2 == 0:
                # For even num_samples, space the values around the original value
                step = (
                    round(abs(value) / (num_samples - 1), 2) if num_samples != 1 else 0
                )
                values = [
                    round(value - step * i, 2)
                    for i in range(num_samples // 2, -num_samples // 2 - 1, -1)
                ]
                # Ensure all remains as int if they started off that way
                if isinstance(value, int):
                    values = [int(val) for val in values]
                grid_params[key] = values
            else:
                # For odd num_samples, include the original value and have equal steps around it
                step = round(abs(value) / (num_samples // 2), 2)
                values = [
                    round(value - step * i, 2)
                    for i in range(num_samples // 2, -num_samples // 2 - 1, -1)
                ]
                # Ensure all remains as int if they started off that way
                if isinstance(value, int):
                    values = [int(val) for val in values]
                grid_params[key] = values
    return grid_params

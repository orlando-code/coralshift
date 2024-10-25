import os
os.environ['OMP_NUM_THREADS'] = "1"
import joblib
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import xgboost as xgb
from coralshift.machine_learning import static_models
import pandas as pd
import numpy as np


def do_search():
    rs_params = static_models.xgb_search_grid(n_trials=5)
    X_train = pd.read_parquet(
        "/maps/rt582/coralshift/data/ml_ready/-20_10/0-01/train_X_0-01_S30-0_S2-0_E140-0_E166-0.parquet")
    y_train = pd.read_parquet(
        "/maps/rt582/coralshift/data/ml_ready/-20_10/0-01/train_y_0-01_S30-0_S2-0_E140-0_E166-0.parquet")
    X_test = pd.read_parquet(
        "/maps/rt582/coralshift/data/ml_ready/-20_10/0-01/test_X_0-01_S30-0_S2-0_E140-0_E166-0.parquet")
    y_test = pd.read_parquet(
        "/maps/rt582/coralshift/data/ml_ready/-20_10/0-01/test_y_0-01_S30-0_S2-0_E140-0_E166-0.parquet")

    X = pd.concat([X_train, X_test], axis=0)
    y = pd.concat([y_train, y_test], axis=0)

    print("initialising model")
    # data = xgb.DMatrix(data=X, label=y)
    xgb_reg = xgb.XGBRegressor(n_jobs=128, seed=42)
    search_object = RandomizedSearchCV(
        estimator=xgb_reg, param_distributions=rs_params, scoring="r2", cv=3, n_iter=100, verbose=10, n_jobs=1)
    search_object.fit(X, y)

    print("Best parameters found: ", search_object.best_params_)
    print("Lowest RMSE found: ", np.sqrt(np.abs(search_object.best_score_)))

    # Save the trained model
    joblib.dump(search_object, '/maps/rt582/coralshift/notebooks/wip/trained_model.joblib')


if __name__ == '__main__':
    do_search()

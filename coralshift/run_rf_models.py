from dask.distributed import Client, LocalCluster
import dask

# local dask cluster
cluster = LocalCluster(n_workers=4)
client = Client(cluster)
client

# TODO: add regressor option
model = XGBRegressor(
    n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8
)

import xgboost as xgb

# Create the XGBoost DMatrices
dtrain = xgb.dask.DaskDMatrix(
    client, dask.array.from_array(X_train), dask.array.from_array(y_train)
)
dtest = xgb.dask.DaskDMatrix(
    client, dask.array.from_array(X_test), dask.array.from_array(y_test)
)

model_type = "brt"
model, data_type = baselines.initialise_model(model_type)
search_grid = baselines.ModelInitializer().get_search_grid(model_type)

param_dict = {
    # 'n_estimators': 100,
    "max_depth": 5,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "random_state": 42,
}

# train the model
output = xgb.dask.train(
    client, param_dict, dtrain, num_boost_round=100, evals=[(dtrain, "train")]
)

# make predictions
y_pred = xgb.dask.predict(client, output, dtest)

from sklearn.metrics import mean_squared_error

mean_squared_error(y_test, y_pred)

import dask_ml.model_selection as dcv
from scipy.stats import expon
from sklearn import svm, datasets

rfr = dcv.RandomizedSearchCV(model, search_grid, n_iter=50, cv=3)
rfr.fit(X_train, y_train)

baselines.evaluate_model(rfr, dfs_list[0], X_train, y_train)

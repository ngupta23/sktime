#%%
from datetime import datetime

now = datetime.now()
print(f"Run Start Time: {now}")

#%%
# Pre Release
import os
import sys
from pprint import pprint

sys.path.append(os.environ["DEV_SKTIME"])

# print(os.environ['DEV_SKTIME'])

#%%
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sktime.forecasting.all import *

#%%
from sktime import show_versions
show_versions()


#%%
y = load_airline()
y = y[:48]

window_length = 18  # How much of previous history to use to train
fh = np.arange(1, 13)  # How much to forecast (from 1 to 12 or 1 year)
step_length = 1  # How much to step the sliding window

#%%
y_train, y_test = temporal_train_test_split(y, test_size=len(fh))
plot_series(y_train, y_test, labels=["y_train", "y_test"])
print(y.shape, y_train.shape[0], y_test.shape[0])
print(y.index)



#%%
# Simple Baseline (with non-stationary data)
# tuning the 'n_estimator' hyperparameter of RandomForestRegressor from scikit-learn
# regressor_param_grid = {"n_estimators": [100, 200, 300]}
regressor_param_grid = {"n_estimators": [200]}
forecaster_param_grid = {"window_length": [12, 15]}

# create a tunnable regressor with GridSearchCV
regressor = GridSearchCV(RandomForestRegressor(), param_grid=regressor_param_grid, verbose=1)
forecaster = ReducedRegressionForecaster(
    regressor, window_length=window_length, strategy="recursive"
)

# cv = SlidingWindowSplitter(initial_window=int(len(y_train) * 0.5))
cv = SlidingWindowSplitter(
    initial_window=20,
    # window_length=10,
    # fh=fh,
    # step_length=step_length,
    start_with_window=True,
)

gscv = ForecastingGridSearchCV(forecaster, cv=cv, param_grid=forecaster_param_grid, verbose=1)

gscv.fit(y_train)
# y_pred = gscv.predict(fh)
# plot_series(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"])
# smape_loss(y_test, y_pred)
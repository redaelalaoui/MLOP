import warnings
from datetime import datetime
from typing import NoReturn

import sklearn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from annexe_functions import standardize_data, inverse_standardization

warnings.filterwarnings('ignore')


class GaussianProcessPrice:
    def __init__(self, selected_kernel: sklearn.gaussian_process.kernels.Product,
                 x_data: list, y_data: list, x_test: list = None, y_test: list = None,
                 standardization: bool = False) -> NoReturn:

        self.selected_kernel = selected_kernel
        self.x_data = x_data
        self.y_data = y_data
        self.x_test = x_test
        self.y_test = y_test
        self.standardization = standardization

        self.y_pred = None
        self.opt_kernel = None
        self.gpr = None
        self.scaler = None

        self.R2 = 0
        self.MAE = 10e5
        self.MSE = 10e5

        self.training_time = 10e5
        self.prediction_time = 10e5

    def train_model(self) -> NoReturn:
        if (self.x_test is None) & (self.y_test is None):
            self.x_data, self.x_test, self.y_data, self.y_test = train_test_split(self.x_data, self.y_data,
                                                                                  test_size=0.2,
                                                                                  random_state=42)

        x_test, y_test = self.x_test, self.y_test
        x_train, y_train = self.x_data, self.y_data

        # Standardization
        if self.standardization:
            scaler, x_train, y_train, x_test, y_test = standardize_data(x_train, y_train, x_test, y_test)
            self.gpr = make_pipeline(StandardScaler(), GaussianProcessRegressor(kernel=self.selected_kernel,
                                                                                random_state=0, normalize_y=True, n_restarts_optimizer=0))
        else:
            self.gpr = GaussianProcessRegressor(kernel=self.selected_kernel, random_state=0, n_restarts_optimizer=0)

        # Trains the model
        t0 = datetime.now()
        self.gpr.fit(x_train, y_train)
        self.training_time = (datetime.now() - t0).total_seconds()

        self.R2 = self.gpr.score(x_train, y_train)

        # Predicts output
        t1 = datetime.now()
        y_pred, sigma_hat = self.gpr.predict(x_test, return_std=True)
        self.prediction_time = (datetime.now() - t1).total_seconds()

        # Inverses standardization
        if self.standardization:
            scaler, x_train, y_train, x_test, y_test, y_pred = \
                inverse_standardization(scaler, x_train, y_train, x_test, y_test, y_pred)

            self.scaler = scaler
            self.opt_kernel = self.gpr[1].kernel_
        else:
            self.opt_kernel = self.gpr.kernel_
        # Stores results

        self.y_pred = y_pred
        self.MAE = mean_absolute_error(y_test, y_pred)
        self.MSE = mean_squared_error(y_test, y_pred)

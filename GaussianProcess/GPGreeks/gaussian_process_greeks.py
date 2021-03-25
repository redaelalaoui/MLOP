import warnings
from typing import NoReturn, Tuple

import numpy as np

from BlackScholes import bsformula
from GaussianProcess.GPPrices.gaussian_process_price import GaussianProcessPrice
from annexe_functions import kernel_derivative
from annexe_functions import k_s_prime_RBF

warnings.filterwarnings('ignore')


class GaussianProcessGreeks:
    def __init__(self, model: GaussianProcessPrice) -> NoReturn:
        # Verifier l'utilité de cette initialisation (les x et y data sont necessaires ?)
        self.gpr = model.gpr
        self.optimal_kernel = model.opt_kernel
        self.x_train = model.x_data
        self.y_train = model.y_data
        self.x_test = model.x_test
        self.y_test = model.y_test

    def delta(self, K: float, r: float, tau: float, sigma: float,
              s_train_lim: Tuple[float, float] = (1., 100.),
              s_test_lim: Tuple[float, float] = (1., 100.)) -> [list, np.ndarray, np.ndarray]:

        s_train = np.linspace(s_train_lim[0], s_train_lim[1], 500)
        s_test = np.linspace(s_test_lim[0], s_test_lim[1], 2000)

        x_train = np.array([[S, K, tau, r, sigma] for S in s_train])
        x_test = np.array([[S, K, tau, r, sigma] for S in s_test])

        y_train = np.array([bsformula(S, K, tau, r, sigma, option='call')[0] for S in x_train[:, 0]])

        # Computes Delta
        f_prime = kernel_derivative(x_train, x_test, y_train, self.optimal_kernel, k_s_prime_RBF, column=0)
        computed_delta = np.array([bsformula(S, K, tau, r, sigma, option='call')[1] for S in x_test[:, 0]])

        return f_prime, computed_delta, s_test

    def vega(self, S: float, K: float, r: float, tau: float,
              sigma_train_lim: Tuple[float, float] = (0.01, 1.),
             sigma_test_lim: Tuple[float, float] = (0.01, 1.)) -> [list, np.ndarray, np.ndarray]:

        sigma_train = np.linspace(sigma_train_lim[0], sigma_train_lim[1], 500)
        sigma_test = np.linspace(sigma_test_lim[0], sigma_test_lim[1], 2000)

        x_train = np.array([[S, K, tau, r, sigma] for sigma in sigma_train])
        x_test = np.array([[S, K, tau, r, sigma] for sigma in sigma_test])

        y_train = np.array([bsformula(S, K, tau, r, sigma, option='call')[0] for sigma in sigma_train])

        # Computes Delta
        f_prime = kernel_derivative(x_train, x_test, y_train, self.optimal_kernel, k_s_prime_RBF, column=-1)
        computed_vega = np.array([bsformula(S, K, tau, r, sigma, option='call')[2] for sigma in sigma_test])

        return f_prime, computed_vega, sigma_test

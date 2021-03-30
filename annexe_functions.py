import warnings
from typing import Callable

import numpy as np
import scipy as sp
import scipy.stats as si
from sklearn.gaussian_process import kernels
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')


def standardize_data(x_train: list, y_train: list, x_test: list, y_test: list, scaler: StandardScaler = None) \
                    -> [StandardScaler, list, list, list, list]:

    new = False
    if scaler is None:
        scaler = StandardScaler()
        new = True
    # Fit on training set only.
    all_data_train = np.append(np.array(x_train), np.array(y_train).reshape(-1, 1), axis=1)
    all_data_test = np.append(np.array(x_test), np.array(y_test).reshape(-1, 1), axis=1)

    if new:
        scaler.fit(all_data_train)

    # Apply transform to both the training set and the test set.
    all_data_train = scaler.transform(all_data_train)
    all_data_test = scaler.transform(all_data_test)

    x_train, y_train = all_data_train[:, :-1], all_data_train[:, -1]
    x_test, y_test = all_data_test[:, :-1], all_data_test[:, -1]

    return scaler, x_train, y_train, x_test, y_test


def inverse_standardization(scaler: StandardScaler, x_train: list, y_train: list,
                            x_test: list, y_test: list, y_pred: list) \
                            -> [StandardScaler, list, list, list, list, list]:

    all_data_train = np.append(np.array(x_train), np.array(y_train).reshape(-1, 1), axis=1)
    all_data_test = np.append(np.array(x_test), np.array(y_test).reshape(-1, 1), axis=1)
    all_data_pred = np.append(np.array(x_test), np.array(y_pred).reshape(-1, 1), axis=1)

    all_data_train = scaler.inverse_transform(all_data_train)
    all_data_test = scaler.inverse_transform(all_data_test)
    all_data_pred = scaler.inverse_transform(all_data_pred)

    x_train, y_train = all_data_train[:, :-1], all_data_train[:, -1]
    x_test, y_test = all_data_test[:, :-1], all_data_test[:, -1]
    x_test, y_pred = all_data_pred[:, :-1], all_data_pred[:, -1]

    return scaler, x_train, y_train, x_test, y_test, y_pred


def k_s_prime_RBF(x_train: np.ndarray, x_test: np.ndarray,
              optimal_kernel: kernels.Product, column: int = 0) -> np.ndarray:

    k_s = optimal_kernel(x_test, x_train)
    l = optimal_kernel.length_scale
    print(l)
    x_train = x_train[:, column].reshape((-1, 1))
    x_test = x_test[:, column].reshape((-1, 1))

    return (x_train.T - x_test) * k_s / l[column] ** 2


def k_s_prime_quadra(x_train: np.ndarray, x_test: np.ndarray,
              optimal_kernel: kernels.Product, column: int = 0) -> np.ndarray:

    k_s = optimal_kernel(x_test, x_train)
    l = optimal_kernel.k2.length_scale
    alpha_quadra = optimal_kernel.k2.alpha

    x_train = x_train[:, column].reshape((-1, 1))
    x_test = x_test[:, column].reshape((-1, 1))

    return (x_train.T - x_test) * k_s / (l ** 2 * (1 + (x_train.T - x_test) ** 2 / (2 * alpha_quadra * l ** 2)))


def kernel_derivative(x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray,
                      optimal_kernel: kernels.Product,
                      k_s_prime: Callable[[np.ndarray, np.ndarray, kernels.Product, int], np.ndarray],
                      column: int = 0, sigma_n: float = 1e-8) -> np.ndarray:

        training_number = len(x_train)

        kernel = optimal_kernel(x_train, x_train)

        k_y = kernel + np.eye(training_number) * sigma_n
        cholesky = sp.linalg.cho_factor(k_y)
        alpha_p = sp.linalg.cho_solve(np.transpose(cholesky), y_train)

        f_prime = np.dot(k_s_prime(x_train, x_test, optimal_kernel, column), alpha_p)

        return f_prime


def compute_price_premium(m, sigma):

    # K: strike price
    # T: time to maturity
    # r: interest rate
    # sigma: volatility of underlying asset

    F_K = np.exp(m*sigma)

    d1 = m + sigma/2
    d2 = m - sigma/2

    call = (np.exp(m*sigma) * si.norm.cdf(d1, 0.0, 1.0) - si.norm.cdf(d2, 0.0, 1.0))
    return (call - np.maximum(F_K - 1, 0))/sigma

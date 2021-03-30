import warnings
from typing import NoReturn, Tuple

import numpy as np

from BlackScholes import bsformula
from GaussianProcess.GPPrices.gaussian_process_price import GaussianProcessPrice
from annexe_functions import compute_price_premium
from annexe_functions import k_s_prime_RBF
from annexe_functions import kernel_derivative

warnings.filterwarnings('ignore')


class GaussianProcessGreeksPricePremium:
    def __init__(self, model_ITM: GaussianProcessPrice, model_OTM: GaussianProcessPrice) -> NoReturn:
        self.optimal_kernel_ITM = model_ITM.opt_kernel
        self.optimal_kernel_OTM = model_OTM.opt_kernel

    def delta(self, K: float, r: float, tau: float, sigma_a: float,
              s_train_lim: Tuple[float, float] = (1., 100.),
              s_test_lim: Tuple[float, float] = (1., 100.)) -> [np.ndarray, np.ndarray]:

        sigma = sigma_a * np.sqrt(tau)

        s_train = np.linspace(s_train_lim[0], s_train_lim[1], 5000)
        s_test = np.linspace(s_test_lim[0], s_test_lim[1], 5000)

        m_train = np.log(s_train * np.exp(r * tau) / K) / sigma
        m_test = np.log(s_test * np.exp(r * tau) / K) / sigma

        m_train_OTM = np.array([m for m in m_train if m <= 0])
        m_train_ITM = np.array([m for m in m_train if m > 0])

        m_test_OTM = np.array([m for m in m_test if m <= 0])
        m_test_ITM = np.array([m for m in m_test if m > 0])

        x_test_OTM = np.array([[m, sigma] for m in m_test_OTM])
        x_train_OTM = np.array([[m, sigma] for m in m_train_OTM])

        x_test_ITM = np.array([[m, sigma] for m in m_test_ITM])
        x_train_ITM = np.array([[m, sigma] for m in m_train_ITM])

        y_train_OTM = compute_price_premium(m_train_OTM, sigma)
        y_train_ITM = compute_price_premium(m_train_ITM, sigma)

        # Computes Delta
        f_prime_ITM = kernel_derivative(x_train_ITM, x_test_ITM, y_train_ITM, self.optimal_kernel_ITM, k_s_prime_RBF, column=0)
        f_prime_OTM = kernel_derivative(x_train_OTM, x_test_OTM, y_train_OTM, self.optimal_kernel_OTM, k_s_prime_RBF, column=0)

        predicted_delta = np.append(np.multiply(np.exp(-m_test_OTM*sigma), f_prime_OTM),
                                    np.multiply(np.exp(-m_test_ITM*sigma), f_prime_ITM) + 1)

        computed_delta = bsformula(s_test, K, tau, r, sigma_a, option='call')[1]

        return predicted_delta, computed_delta, s_test

    def vega(self, S: float, K: float, r: float, tau: float,
              sigma_train_lim: Tuple[float, float] = (0.03, 1),
             sigma_test_lim: Tuple[float, float] = (0.03, 1)) -> [list, np.ndarray, np.ndarray]:

        sigma_a_train = np.linspace(sigma_train_lim[0], sigma_train_lim[1], 2000)
        sigma_a_test = np.linspace(sigma_test_lim[0], sigma_test_lim[1], 2000)

        sigma_train = sigma_a_train * np.sqrt(tau)
        sigma_test = sigma_a_test * np.sqrt(tau)

        m_train = np.log(S * np.exp(r * tau) / K) / sigma_train
        m_test = np.log(S * np.exp(r * tau) / K) / sigma_test

        m_train_OTM, m_train_ITM, m_test_OTM, m_test_ITM = [], [], [], []
        sigma_train_OTM, sigma_train_ITM, sigma_test_OTM, sigma_test_ITM = [], [], [], []
        sigma_a_train_OTM, sigma_a_train_ITM, sigma_a_test_OTM, sigma_a_test_ITM = [], [], [], []

        for i in range(len(m_train)):
            if m_train[i] < 0:
                m_train_OTM.append(m_train[i])
                sigma_train_OTM.append(sigma_train[i])
                sigma_a_train_OTM.append(sigma_a_train[i])
            else:
                m_train_ITM.append(m_train[i])
                sigma_train_ITM.append(sigma_train[i])
                sigma_a_train_ITM.append(sigma_a_train[i])

        for i in range(len(m_test)):
            if m_test[i] < 0:
                m_test_OTM.append(m_test[i])
                sigma_test_OTM.append(sigma_test[i])
                sigma_a_test_OTM.append(sigma_a_test[i])
            else:
                m_test_ITM.append(m_test[i])
                sigma_test_ITM.append(sigma_test[i])
                sigma_a_test_ITM.append(sigma_a_test[i])

        x_test_OTM = np.array([[m_test_OTM[i], sigma_test_OTM[i]] for i in range(len(m_test_OTM))])
        x_train_OTM = np.array([[m_train_OTM[i], sigma_train_OTM[i]] for i in range(len(m_train_OTM))])

        x_test_ITM = np.array([[m_test_ITM[i], sigma_test_ITM[i]] for i in range(len(m_test_ITM))])
        x_train_ITM = np.array([[m_train_ITM[i], sigma_train_ITM[i]] for i in range(len(m_train_ITM))])

        # Computes Vega
        f_prime_m_OTM, f_prime_sigma_OTM, f_prime_m_ITM, f_prime_sigma_ITM = [], [], [], []
        y_train_OTM, y_test_OTM, y_train_ITM, y_test_ITM = [], [], [], []

        if (len(m_test_OTM) != 0) & (len(m_train_OTM) != 0):
            y_train_OTM = compute_price_premium(x_train_OTM[:, 0], x_train_OTM[:, 1])
            y_test_OTM = compute_price_premium(x_test_OTM[:, 0], x_test_OTM[:, 1])
            f_prime_m_OTM = kernel_derivative(x_train_OTM, x_test_OTM, y_train_OTM,
                                              self.optimal_kernel_OTM, k_s_prime_RBF, column=0)

            f_prime_sigma_OTM = kernel_derivative(x_train_OTM, x_test_OTM, y_train_OTM,
                                                  self.optimal_kernel_OTM, k_s_prime_RBF, column=-1)


        if (len(m_test_ITM) != 0) & (len(m_train_ITM) != 0):
            y_train_ITM = compute_price_premium(x_train_ITM[:, 0], x_train_ITM[:, 1])
            y_test_ITM = compute_price_premium(x_test_ITM[:, 0], x_test_ITM[:, 1])

            f_prime_m_ITM = kernel_derivative(x_train_ITM, x_test_ITM, y_train_ITM,
                                              self.optimal_kernel_ITM, k_s_prime_RBF, column=0)

            f_prime_sigma_ITM = kernel_derivative(x_train_ITM, x_test_ITM, y_train_ITM,
                                                  self.optimal_kernel_ITM, k_s_prime_RBF, column=-1)

        f_prime_sigma = np.append(f_prime_sigma_OTM, f_prime_sigma_ITM)
        f_prime_m = np.append(f_prime_m_OTM, f_prime_m_ITM)
        y_test = np.append(y_test_OTM, y_test_ITM)

        vega = (K * np.exp(-r * tau) * ((y_test + f_prime_sigma) * np.sqrt(tau) -
                                        np.multiply(m_test / sigma_a_test, f_prime_m)))

        computed_vega = bsformula(S, K, tau, r, sigma_a_test, option='call')[2]

        return vega, computed_vega, sigma_a_test

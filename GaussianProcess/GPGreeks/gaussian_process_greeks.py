import warnings
from typing import NoReturn, Tuple

import numpy as np

from BlackScholes import bsformula
from GaussianProcess.GPPrices.gaussian_process_price import GaussianProcessPrice
from annexe_functions import compute_price_premium
from annexe_functions import k_s_prime_RBF, standardize_data, inverse_standardization
from annexe_functions import kernel_derivative

warnings.filterwarnings('ignore')


class GaussianProcessGreeks:
    def __init__(self, model: GaussianProcessPrice, model_ITM: GaussianProcessPrice, model_OTM: GaussianProcessPrice) -> NoReturn:
        if model is None:
            self.optimal_kernel_ITM = model_ITM.opt_kernel
            self.optimal_kernel_OTM = model_OTM.opt_kernel
            self.gpr_ITM = model_ITM.gpr
            self.gpr_OTM = model_OTM.gpr
            if model_ITM.standardization:
                self.scaler_ITM = model_ITM.scaler
                self.scaler_OTM = model_OTM.scaler
        else:
            self.optimal_kernel = model.opt_kernel
            self.gpr = model.gpr
            if model.standardization:
                self.scaler = model.scaler

    def function(self, stand: bool, naive: bool, price: bool, K: float, r: float, tau: float, sigma_a: float,
                 s_test_lim: Tuple[float, float] = (1., 100.)) -> [list, np.ndarray, np.ndarray]:

        s_test = np.linspace(s_test_lim[0], s_test_lim[1], 1000)

        sigma = sigma_a * np.sqrt(tau)
        m_test = np.log(s_test * np.exp(r * tau) / K) / sigma

        if price:
            # Naive variables and call price

            x_test = np.array([[S, K, tau, r, sigma_a] for S in s_test])
            y_test = np.array([bsformula(S, K, tau, r, sigma_a, option='call')[0] for S in s_test])

            if stand:
                _, _, _, x_test, _ = standardize_data(x_test, y_test, x_test, y_test, self.scaler)

            y_pred, sigma_hat = self.gpr.predict(x_test, return_std=True)

            if stand:
                _, _, _, _, _, y_pred = inverse_standardization(self.scaler, x_test, y_test, x_test, y_test, y_pred)

            return m_test, y_test, y_pred

        else:
            m_test_OTM = np.array([m for m in m_test if m <= 0])
            m_test_ITM = np.array([m for m in m_test if m > 0])

            if naive:
                s_test_OTM = np.array([s for s in s_test if np.log(s * np.exp(r * tau) / K) <= 0])
                s_test_ITM = np.array([s for s in s_test if np.log(s * np.exp(r * tau) / K) > 0])

                x_test_OTM = np.array([[s, K, tau, r, sigma_a] for s in s_test_OTM])
                x_test_ITM = np.array([[s, K, tau, r, sigma_a] for s in s_test_ITM])
            else:
                x_test_OTM = np.array([[m, sigma] for m in m_test_OTM])
                x_test_ITM = np.array([[m, sigma] for m in m_test_ITM])

            # for all pp
            y_test_OTM = compute_price_premium(m_test_OTM, sigma)
            y_test_ITM = compute_price_premium(m_test_ITM, sigma)

            if stand:
                _, _, _, x_test_OTM, _ = standardize_data(x_test_OTM, y_test_OTM, x_test_OTM, y_test_OTM, self.scaler_OTM)
                _, _, _, x_test_ITM, _ = standardize_data(x_test_ITM, y_test_ITM, x_test_ITM, y_test_ITM, self.scaler_ITM)

            y_pred_OTM, _ = self.gpr_OTM.predict(x_test_OTM, return_std=True)
            y_pred_ITM, _ = self.gpr_ITM.predict(x_test_ITM, return_std=True)

            if stand:
                _, _, _, _, _, y_pred_OTM = inverse_standardization(self.scaler_OTM, x_test_OTM, y_test_OTM, x_test_OTM, y_test_OTM, y_pred_OTM)
                _, _, _, _, _, y_pred_ITM = inverse_standardization(self.scaler_ITM, x_test_ITM, y_test_ITM, x_test_ITM, y_test_ITM, y_pred_ITM)

            return np.append(m_test_OTM, m_test_ITM), np.append(y_test_OTM, y_test_ITM), np.append(y_pred_OTM, y_pred_ITM)

    def delta(self, naive: bool, price: bool, K: float, r: float, tau: float, sigma_a: float,
              s_train_lim: Tuple[float, float] = (1., 100.),
              s_test_lim: Tuple[float, float] = (1., 100.)) -> [list, np.ndarray, np.ndarray]:

        s_train = np.linspace(s_train_lim[0], s_train_lim[1], 1000)
        s_test = np.linspace(s_test_lim[0], s_test_lim[1], 1000)

        if price:
            # Naive variables and call price
            x_train = np.array([[S, K, tau, r, sigma_a] for S in s_train])
            x_test = np.array([[S, K, tau, r, sigma_a] for S in s_test])

            y_train = np.array([bsformula(S, K, tau, r, sigma_a, option='call')[0] for S in x_train[:, 0]])

            # Naive and price
            f_prime = kernel_derivative(x_train, x_test, y_train, self.optimal_kernel, k_s_prime_RBF, column=0)

        else:
            sigma = sigma_a * np.sqrt(tau)
            m_train = np.log(s_train * np.exp(r * tau) / K) / sigma
            m_test = np.log(s_test * np.exp(r * tau) / K) / sigma

            m_train_OTM = np.array([m for m in m_train if m <= 0])
            m_train_ITM = np.array([m for m in m_train if m > 0])

            if naive:
                # Naive variables and pp
                s_train_OTM = np.array([s for s in s_train if np.log(s * np.exp(r * tau) / K) <= 0])
                s_train_ITM = np.array([s for s in s_train if np.log(s * np.exp(r * tau) / K) >= 0])

                s_test_OTM = np.array([s for s in s_test if np.log(s * np.exp(r * tau) / K) <= 0])
                s_test_ITM = np.array([s for s in s_test if np.log(s * np.exp(r * tau) / K) > 0])

                x_test_OTM = np.array([[s, K, tau, r, sigma_a] for s in s_test_OTM])
                x_train_OTM = np.array([[s, K, tau, r, sigma_a] for s in s_train_OTM])

                x_test_ITM = np.array([[s, K, tau, r, sigma_a] for s in s_test_ITM])
                x_train_ITM = np.array([[s, K, tau, r, sigma_a] for s in s_train_ITM])

                # for all pp
                y_train_OTM = compute_price_premium(m_train_OTM, sigma)
                y_train_ITM = compute_price_premium(m_train_ITM, sigma)

                # Naive and pp
                f_prime_ITM = kernel_derivative(x_train_ITM, x_test_ITM, y_train_ITM, self.optimal_kernel_ITM,
                                                k_s_prime_RBF, column=0)
                f_prime_OTM = kernel_derivative(x_train_OTM, x_test_OTM, y_train_OTM, self.optimal_kernel_OTM,
                                                k_s_prime_RBF, column=0)

                f_prime = np.append(K * sigma * np.exp(-r * tau) * f_prime_OTM,
                                            K * sigma * np.exp(-r * tau) * f_prime_ITM + 1)

            else:
                # Reduced variables and pp
                m_test_OTM = np.array([m for m in m_test if m <= 0])
                m_test_ITM = np.array([m for m in m_test if m > 0])

                x_test_OTM = np.array([[m, sigma] for m in m_test_OTM])
                x_train_OTM = np.array([[m, sigma] for m in m_train_OTM])

                x_test_ITM = np.array([[m, sigma] for m in m_test_ITM])
                x_train_ITM = np.array([[m, sigma] for m in m_train_ITM])

                # for all pp
                y_train_OTM = compute_price_premium(m_train_OTM, sigma)
                y_train_ITM = compute_price_premium(m_train_ITM, sigma)

                # Reduced and pp
                f_prime_ITM = kernel_derivative(x_train_ITM, x_test_ITM, y_train_ITM, self.optimal_kernel_ITM,
                                                k_s_prime_RBF, column=0)
                f_prime_OTM = kernel_derivative(x_train_OTM, x_test_OTM, y_train_OTM, self.optimal_kernel_OTM,
                                                k_s_prime_RBF, column=0)

                f_prime = np.append(np.multiply(np.exp(-m_test_OTM * sigma), f_prime_OTM),
                                    np.multiply(np.exp(-m_test_ITM * sigma), f_prime_ITM) + 1)
        # for all
        computed_delta = np.array([bsformula(S, K, tau, r, sigma_a, option='call')[1] for S in s_test])

        return f_prime, computed_delta, s_test

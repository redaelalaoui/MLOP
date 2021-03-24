from typing import NoReturn, Tuple

import numpy as np
import pandas as pd

from BlackScholes import bsformula
from annexe_functions import compute_price_premium


def naive_approach(spot_range: Tuple[float, float], strike_range: Tuple[float, float],
                   rate_range: Tuple[float, float], maturity_range: Tuple[float, float],
                   vol_range: Tuple[float, float], data_size: int = 5000, kind_of_data: str = 'train') -> NoReturn:

    # Initialization
    filepath = "C:/Users/Edgelab/PycharmProjects/pythonProject/VanillaOptions/Data/"
    n_dis = 1000

    all_spot = np.linspace(spot_range[0], spot_range[1], n_dis)
    all_strikes = np.linspace(strike_range[0], strike_range[1], n_dis)
    all_rates = np.linspace(rate_range[0], rate_range[1], n_dis)
    all_maturities = np.linspace(maturity_range[0], maturity_range[1], n_dis)
    all_volatilities = np.linspace(vol_range[0], vol_range[1], n_dis)

    # Compute data
    x_data = [[all_spot[np.random.randint(n_dis)], all_strikes[np.random.randint(n_dis)],
               all_maturities[np.random.randint(n_dis)], all_rates[np.random.randint(n_dis)],
               all_volatilities[np.random.randint(n_dis)]] for _ in range(data_size)]

    y_data = [bsformula(x_row[0], x_row[1], x_row[2], x_row[3], x_row[4], option='call')[0] for x_row in x_data]

    # Store data
    df = pd.DataFrame(x_data)
    df.columns = ['S', 'K', 'tau', 'r', 'sigma']
    df['Call price'] = y_data
    df.to_csv(filepath + f"{kind_of_data}_naive_approach.csv", sep=',', index=False)
    print(f'Black and Scholes {kind_of_data} dataset: created.')


# PRICE PREMIUM APPROACH
def price_premium_data(m_range: Tuple[float, float], sigma_range: Tuple[float, float],
                       kind_of_data: str = 'train') -> NoReturn:
    # Initialization
    filepath = "C:/Users/Edgelab/PycharmProjects/pythonProject/VanillaOptions/Data/"
    x_data, y_data = [], []

    list_m = np.linspace(m_range[0], m_range[1], 70)
    list_sigma = np.linspace(sigma_range[0], sigma_range[1], 70)

    # Compute data
    for m in list_m:
        for sigma in list_sigma:
            x_data.append([m, sigma])
            y_data.append(compute_price_premium(m, sigma))

    # Store data
    df = pd.DataFrame(x_data)
    df.columns = ['m', 'sigma']
    df['Price Premium'] = y_data
    df.to_csv(filepath + f"{kind_of_data}_price_premium.csv", sep=',', index=False)
    print(f'Price premium {kind_of_data} dataset: created.')


if __name__ == "__main__":

    # NAIVE APPROACH
    # Train set
    naive_approach(spot_range=(40, 140), strike_range=(60, 110),
                   rate_range=(0.01, 0.1), maturity_range=(1/12, 3),
                   vol_range=(0.05, 0.15), data_size=5000, kind_of_data='train')
    # Test set
    naive_approach(spot_range=(50, 120), strike_range=(60, 110),
                   rate_range=(0.01, 0.1), maturity_range=(1/12, 3),
                   vol_range=(0.05, 0.15), data_size=5000, kind_of_data='test')

    # PRICE PREMIUM APPROACH
    # Train set
    price_premium_data(m_range=(-2., 2.), sigma_range=(0.1, 0.9), kind_of_data='train')

    # Test set
    price_premium_data(m_range=(-2., 2.), sigma_range=(0.1, 0.9), kind_of_data='test')


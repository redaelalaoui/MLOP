from typing import NoReturn, Tuple

import numpy as np
import pandas as pd

from BlackScholes import bsformula
from annexe_functions import compute_price_premium


def black_scholes_datasets(spot_range: Tuple[float, float], strike_range: Tuple[float, float],
                           rate_range: Tuple[float, float], maturity_range: Tuple[float, float],
                           vol_range: Tuple[float, float], data_size: int = 5000, kind_of_data: str = 'train') -> NoReturn:

    # Initialization
    filepath = "C:/Users/Edgelab/PycharmProjects/VanillaOptions/Data/"
    n_dis = 5

    all_spot = np.linspace(spot_range[0], spot_range[1], n_dis)
    all_strikes = np.linspace(strike_range[0], strike_range[1], n_dis)
    all_rates = np.linspace(rate_range[0], rate_range[1], n_dis)
    all_maturities = np.linspace(maturity_range[0], maturity_range[1], n_dis)
    all_volatilities = np.linspace(vol_range[0], vol_range[1], n_dis)

    # Inputs
    # Naive variables
    x_naive = np.array([[all_spot[np.random.randint(n_dis)], all_strikes[np.random.randint(n_dis)],
                       all_maturities[np.random.randint(n_dis)], all_rates[np.random.randint(n_dis)],
                       all_volatilities[np.random.randint(n_dis)]] for _ in range(data_size)])

    # Reduced variables
    sigma = x_naive[:, 4] * np.sqrt(x_naive[:, 2])
    m = np.log(x_naive[:, 0] * np.exp(x_naive[:, 3] * x_naive[:, 2])/x_naive[:, 1])/sigma

    x_reduced = np.array([[m[i], sigma[i]] for i in range(len(m))])

    # Outputs
    # Call price
    y_price = [bsformula(x_row[0], x_row[1], x_row[2], x_row[3], x_row[4], option='call')[0] for x_row in x_naive]

    # Price premium
    y_pp = compute_price_premium(m, sigma)

    # Input: Naive variables
    # Output: Price premium
    df = pd.DataFrame(x_naive)
    df.columns = ['S', 'K', 'tau', 'r', 'sigma']
    df['Price Premium'] = y_pp
    df.to_csv(filepath + f"{kind_of_data}_naive_pp.csv", sep=',', index=False)

    # Input: Reduced variables
    # Output: Price premium
    df = pd.DataFrame(x_reduced)
    df.columns = ['m', 'sigma']
    df['Price Premium'] = y_pp
    df.to_csv(filepath + f"{kind_of_data}_reduced_pp.csv", sep=',', index=False)

    # Input: Naive variables
    # Output: Call price
    df = pd.DataFrame(x_naive)
    df.columns = ['S', 'K', 'tau', 'r', 'sigma']
    df['Call price'] = y_price
    df.to_csv(filepath + f"{kind_of_data}_naive_price.csv", sep=',', index=False)

    # Input: Reduced variables
    # Output: Call price
    df = pd.DataFrame(x_reduced)
    df.columns = ['m', 'sigma']
    df['Call price'] = y_price
    df.to_csv(filepath + f"{kind_of_data}_reduced_price.csv", sep=',', index=False)


if __name__ == "__main__":

    # TWO APPROACHES
    # Train set
    two_approaches(spot_range=(40, 140), strike_range=(95, 105),
                   rate_range=(0.001, 0.15), maturity_range=(1/12, 4),
                   vol_range=(0.03, 0.4), data_size=5000, kind_of_data='train')

    # Test set
    two_approaches(spot_range=(40, 140), strike_range=(95, 105),
                   rate_range=(0.001, 0.15), maturity_range=(1/12, 4),
                   vol_range=(0.03, 0.4), data_size=5000, kind_of_data='test')

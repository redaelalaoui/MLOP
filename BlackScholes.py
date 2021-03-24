import math

import numpy as np
import scipy.stats as si


def bsformula(S, K, T, r, sigma, option='call'):

    # S: spot price
    # K: strike price
    # T: time to maturity
    # r: interest rate
    # sigma: volatility of underlying asset

    assert option == 'call' or option == 'put'

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    if option == 'call':
        cp = 1
    else:
        cp = -1
    
    price = (cp * S * si.norm.cdf(cp * d1, 0.0, 1.0) - cp * K * np.exp(-r * T) * si.norm.cdf(cp * d2, 0.0, 1.0))
    delta = cp * si.norm.cdf(cp * d1)
    vega = S*np.sqrt(T)*si.norm.pdf(d1)

    return price, delta, vega


if __name__ == "__main__":
     call_price = bsformula(100.0, 90.0, 2, 0.04, 0.2)

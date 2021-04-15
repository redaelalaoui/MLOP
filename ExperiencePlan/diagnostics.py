import warnings
from typing import NoReturn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from GaussianProcess.GPGreeks.general_greeks import GaussianProcessGreeks

warnings.filterwarnings('ignore')


# Plot the MSE and the training time vs. training size
def diagnostics_metrics(title: str, df_experiments: pd.DataFrame, hue: str = None, style: str = None) -> NoReturn:
    sns.set_style("whitegrid")
    fig, axs = plt.subplots(figsize=(10, 4), ncols=2)
    # 1 - MSE
    sns.lineplot(x='Training size', y='MSE', data=df_experiments, ax=axs[0], hue=hue,
                 style=style).set(xscale="log", yscale="log")

    # 2 - Training time
    sns.lineplot(x='Training size', y='Training time', data=df_experiments, ax=axs[1], hue=hue,
                 style=style).set(xscale="log", yscale="log")

    plt.savefig(f"Results/diagnostics/{title.lower().replace(' ', '_')}_metrics.png")
    plt.show()


# Plot the Call price / Price premium (and the error) vs. moneyness
def diagnostic_function(price_vs_pp: str, title: str, tau: float = 2., sigma_tau: float = 0.25,
                        stand: bool = False, naive: bool = False, price: bool = False,
                        model: GaussianProcessPrice = None,
                        model_ITM: GaussianProcessPrice = None,
                        model_OTM: GaussianProcessPrice = None) -> NoReturn:

    m_test, y_test, y_pred = GaussianProcessGreeks(model=model, model_ITM=model_ITM, model_OTM=model_OTM). \
        function(stand=stand, naive=naive, price=price, K=100, r=0.05, tau=tau,
                 sigma_a=sigma_tau/np.sqrt(tau), s_test_lim=(55., 145.))

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    axs[0].plot(m_test, y_test, '-g', label='Analytical formula')
    axs[0].plot(m_test, y_pred, '--r', label='Gaussian Process')
    axs[0].set_title(title+' (for $\sigma_{\\tau}$='+str(sigma_tau)+')', size=16)
    axs[0].set_ylabel(ylabel=price_vs_pp, fontsize=16)
    axs[0].set_xlabel(xlabel='m', fontsize=16)
    axs[0].legend(loc='best', prop={'size': 12})
    axs[0].set_yscale('log')

    axs[1].plot(m_test, abs(y_pred - y_test), 'r', label='error')
    axs[1].set_title(f'Error on {price_vs_pp}'+' (for $\sigma_{\\tau}$='+str(sigma_tau)+')', size=16)
    axs[1].set_ylabel(ylabel='error', fontsize=16)
    axs[1].set_xlabel(xlabel='m', fontsize=16)
    axs[1].legend(loc='best', prop={'size': 12})
    axs[1].set_yscale('log')

    plt.savefig(f"Results/diagnostics/{title.lower().replace(' ', '_')}_{str(sigma_tau).replace('.', '_')}.png")
    plt.show()


# Plot the Delta vs. moneyness (only for RBF kernel)
def diagnostics_greeks(title: str, naive: bool = False, price: bool = False,
                       tau: float = 2., sigma_tau: float = 0.25,
                       model: GaussianProcessPrice = None,
                       model_ITM: GaussianProcessPrice = None,
                       model_OTM: GaussianProcessPrice = None) -> NoReturn:

    if naive & price:
        delta, computed_delta, s_range = GaussianProcessGreeks(model=model, model_ITM=model_ITM, model_OTM=model_OTM).\
            delta(naive, price, 100, 0.05, tau, sigma_tau/np.sqrt(tau), s_train_lim=(50, 135), s_test_lim=(50, 135))

    elif naive & (not price):
        # 4 - delta vs m
        delta, computed_delta, s_range = GaussianProcessGreeks(model=None, model_ITM=model_ITM, model_OTM=model_OTM).\
            delta(naive, price, 100, 0.05, tau, sigma_tau/np.sqrt(tau), s_train_lim=(50, 135), s_test_lim=(50, 135))

    elif (not naive) & price:
        print("not for this combinaison")
        s_range = None
        computed_delta = None
        delta = None

    else:
        delta, computed_delta, s_range = GaussianProcessGreeks(model=None, model_ITM=model_ITM, model_OTM=model_OTM).\
            delta(naive, price, 100, 0.05, tau, sigma_tau/np.sqrt(tau), s_train_lim=(50, 135), s_test_lim=(50, 135))

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    axs[0].plot(s_range, computed_delta, 'b', label='Analytical formula')
    axs[0].plot(s_range, delta, '--r', label='Gaussian process')
    axs[0].set_title('Delta (for $\sigma_{\\tau}$='+str(sigma_tau)+')', size=16)
    axs[0].set_ylabel(ylabel='$\Delta$', fontsize=16)
    axs[0].set_xlabel(xlabel='S', fontsize=16)
    axs[0].legend(loc='best', prop={'size': 12})

    axs[1].plot(s_range, abs(computed_delta - delta), 'r', label='Error')
    axs[1].set_title('Error in Delta (for $\sigma_{\\tau}$='+str(sigma_tau)+')', size=16)
    axs[1].set_ylabel(ylabel='$err_{\Delta}$', fontsize=16)
    axs[1].set_xlabel(xlabel='S', fontsize=16)
    axs[1].legend(loc='best', prop={'size': 12})
    axs[1].set_yscale('log')

    plt.savefig(f"Results/diagnostics/{title.lower().replace(' ', '_')}_{str(sigma_tau).replace('.', '_')}_delta.png")
    plt.show()

import warnings
from typing import Tuple

import numpy as np
import pandas as pd
import sklearn
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic

from GaussianProcess.GPPrices.gaussian_process_price import GaussianProcessPrice

warnings.filterwarnings('ignore')


# Tests all possibilities
def experience_plan(data_sizes: list, naive_vs_reduced: Tuple[bool, bool] = (True, True),
                    price_vs_pp: Tuple[bool, bool] = (True, True),
                    stand_vs_nostand: Tuple[bool, bool] = (True, True),
                    iso_vs_aniso: Tuple[bool, bool] = (True, True),
                    kernels: Tuple[bool, bool, bool] = (True, True, True)) \
        -> [pd.DataFrame, GaussianProcessPrice, GaussianProcessPrice]:

    # 0 - Initialization
    filepath = "C:/Users/Edgelab/PycharmProjects/VanillaOptions/Data/"

    performances = pd.DataFrame(columns=['Kernel', 'Optimal kernel', 'R2', 'Training time', 'Prediction time',
                                         'MAE', 'MSE', 'Training size', 'Standaridization',
                                         'input', 'target', 'isotropy'])

    for data_size in data_sizes:

        # 1 - Pre-processing
        # 1.1 - Select the data
        selected_data = []
        if naive_vs_reduced[0]:
            if price_vs_pp[0]:
                data_naive_price = pd.read_csv(filepath + 'train_naive_price.csv').sample(data_size)
                selected_data.append((data_naive_price, 'naive', 'price'))
            if price_vs_pp[1]:
                data_naive_pp = pd.read_csv(filepath + 'train_naive_pp.csv').sample(data_size)
                selected_data.append((data_naive_pp, 'naive', 'pp'))

        if naive_vs_reduced[1]:
            if price_vs_pp[0]:
                data_reduced_price = pd.read_csv(filepath + 'train_reduced_price.csv').sample(data_size)
                selected_data.append((data_reduced_price, 'reduced', 'price'))
            if price_vs_pp[1]:
                data_reduced_pp = pd.read_csv(filepath + 'train_reduced_pp.csv').sample(data_size)
                selected_data.append((data_reduced_pp, 'reduced', 'pp'))

        # 1.2 - Standardization and Isotropy
        standardization = []
        if stand_vs_nostand[0]:
            standardization.append(True)
        if stand_vs_nostand[1]:
            standardization.append(False)

        isotropy = []
        if iso_vs_aniso[0]:
            isotropy.append(True)
        if iso_vs_aniso[1]:
            isotropy.append(False)

        # 1.3 - Select the kernels
        selected_kernels = []
        if kernels[0]:
            sqe_kernel = lambda l: RBF(length_scale=l, length_scale_bounds=(1e-12, 1000000.0))
            selected_kernels.append((sqe_kernel, 'RBF'))

        if kernels[1]:
            quad_kernel = lambda l: RationalQuadratic(length_scale=1.5, alpha=1.5,
                                                          length_scale_bounds=(1e-12, 1000000.0),
                                                          alpha_bounds=(1e-12, 1000000.0))
            selected_kernels.append((quad_kernel, 'RQ'))

        if kernels[2]:
            mat_kernel = lambda l: Matern(length_scale=l, length_scale_bounds=(1e-12, 1000000.0), nu=2.5)
            selected_kernels.append((mat_kernel, 'Matern'))

        # 2 - Model training
        for iso in isotropy:
            for data in selected_data:
                # MODIF : 1,5 en 2
                l = 0.5
                if data[1] == 'naive':
                    if not iso:
                        l = [0.5, 1.5, 5, 1, 15]
                else:
                    if not iso:
                        l = [0.3, 6]

                for stand in standardization:
                    for kernel in selected_kernels:
                        performances, model_1, model_2 = train_the_model(kernel, l, data[0], stand, performances,
                                                             data[2], data[1], data_size, iso)

    return performances, model_1, model_2


# Train the model
def train_the_model(kernel: sklearn.gaussian_process.kernels, l: list, data: pd.DataFrame, stand: bool,
                    performances: pd.DataFrame, price_vs_pp: str, naive_vs_reduced: str, data_size: list, iso: bool)\
        -> [pd.DataFrame, GaussianProcessPrice, GaussianProcessPrice]:

    if price_vs_pp == 'price':
        y_train = data['Call price']
        if naive_vs_reduced == 'naive':
            X_train = data[['S', 'K', 'tau', 'r', 'sigma']]
        else:
            X_train = data[['m', 'sigma']]

        model = GaussianProcessPrice(kernel[0](l), X_train, y_train, standardization=stand)
        model.train_model()

        performances = performances.append(
            {'Kernel': kernel[1], 'Optimal kernel': model.opt_kernel, 'R2': model.R2,
             'Training time': model.training_time, 'Prediction time': model.prediction_time,
             'MAE': model.MAE, 'MSE': model.MSE, 'Training size': data_size,
             'Standaridization': stand, 'input': naive_vs_reduced, 'target': price_vs_pp,
             'isotropy': iso}, ignore_index=True)

        return performances, model, None

    elif (price_vs_pp == 'pp') & (naive_vs_reduced == 'reduced'):
        data_train_ITM = data[data['m'] >= 0]
        data_train_OTM = data[data['m'] < 0]

        x_train_ITM = data_train_ITM[['m', 'sigma']]
        y_train_ITM = data_train_ITM['Price Premium']

        x_train_OTM = data_train_OTM[['m', 'sigma']]
        y_train_OTM = data_train_OTM['Price Premium']

        model_ITM = GaussianProcessPrice(kernel[0](l), x_train_ITM, y_train_ITM, standardization=stand)
        model_ITM.train_model()

        performances = performances.append(
            {'Kernel': kernel[1], 'Optimal kernel': model_ITM.opt_kernel, 'R2': model_ITM.R2,
             'Training time': model_ITM.training_time, 'Prediction time': model_ITM.prediction_time,
             'MAE': model_ITM.MAE, 'MSE': model_ITM.MSE, 'Training size': data_size,
             'Standaridization': stand, 'input': naive_vs_reduced, 'target': price_vs_pp + ' ITM',
             'isotropy': iso}, ignore_index=True)

        model_OTM = GaussianProcessPrice(kernel[0](l), x_train_OTM, y_train_OTM, standardization=stand)
        model_OTM.train_model()

        performances = performances.append(
            {'Kernel': kernel[1], 'Optimal kernel': model_OTM.opt_kernel, 'R2': model_OTM.R2,
             'Training time': model_OTM.training_time, 'Prediction time': model_OTM.prediction_time,
             'MAE': model_OTM.MAE, 'MSE': model_OTM.MSE, 'Training size': data_size,
             'Standaridization': stand, 'input': naive_vs_reduced, 'target': price_vs_pp + ' OTM',
             'isotropy': iso}, ignore_index=True)

        return performances, model_ITM, model_OTM

    else:
        data['m'] = np.log(np.multiply(np.array(data['S']), np.exp(np.multiply(np.array(data['r']), np.array(data['tau']))))/np.array(data['K']))

        data_train_ITM = data[data['m'] >= 0]
        data_train_OTM = data[data['m'] < 0]

        x_train_ITM = data_train_ITM[['S', 'K', 'tau', 'r', 'sigma']]
        y_train_ITM = data_train_ITM['Price Premium']

        x_train_OTM = data_train_OTM[['S', 'K', 'tau', 'r', 'sigma']]
        y_train_OTM = data_train_OTM['Price Premium']

        model_ITM = GaussianProcessPrice(kernel[0](l), x_train_ITM, y_train_ITM, standardization=stand)
        model_ITM.train_model()

        performances = performances.append(
            {'Kernel': kernel[1], 'Optimal kernel': model_ITM.opt_kernel, 'R2': model_ITM.R2,
             'Training time': model_ITM.training_time, 'Prediction time': model_ITM.prediction_time,
             'MAE': model_ITM.MAE, 'MSE': model_ITM.MSE, 'Training size': data_size,
             'Standaridization': stand, 'input': naive_vs_reduced, 'target': price_vs_pp + ' ITM',
             'isotropy': iso}, ignore_index=True)

        model_OTM = GaussianProcessPrice(kernel[0](l), x_train_OTM, y_train_OTM, standardization=stand)
        model_OTM.train_model()

        performances = performances.append(
            {'Kernel': kernel[1], 'Optimal kernel': model_OTM.opt_kernel, 'R2': model_OTM.R2,
             'Training time': model_OTM.training_time, 'Prediction time': model_OTM.prediction_time,
             'MAE': model_OTM.MAE, 'MSE': model_OTM.MSE, 'Training size': data_size,
             'Standaridization': stand, 'input': naive_vs_reduced, 'target': price_vs_pp + ' OTM',
             'isotropy': iso}, ignore_index=True)

        return performances, model_ITM, model_OTM

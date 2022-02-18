"""
To replicate coverage results in section 4.1 
in "Learning Pareto-Efficient Decisions with Confidence".
"""

import numpy as np
import synthetic_data
import torch
import random
import helper_weighted_cqr

from cqr.cqr_weighter import BaseWeighter
from train_and_evaluate import TrainAndEvaluate

from cqr import helper
from cqr.nc import RegressorNc
from cqr.nc import WeightedQuantileRegErrFunc

n_mc = 500
alphas = (0.05, 0.10, 0.15, 0.20, 0.25)

rho = -0.2

n_train = 1000
n_test = 100
in_shape = 2  # Dimensions Z + X
n_x = 5

# %% Conformal intervals (See QuantileForestRegressorAdapter class in helper.py)
# Parameters random forest
n_estimators = 1000  # number of trees in the forest
min_samples_leaf = 1  # minimum number of samples at a leaf node
max_features = 'auto'  # number of features to consider when looking for the best split
cv_qforest = False  # is cross-validation used to tune the quantile levels?

# Define the QRF's parameters
params_qforest = dict()
params_qforest["n_estimators"] = n_estimators
params_qforest["min_samples_leaf"] = min_samples_leaf
params_qforest["max_features"] = max_features
params_qforest["CV"] = cv_qforest

# Set seed
shift = 123
seeds = np.arange(shift + 1, shift + n_mc + 1)

coverage_y1 = np.zeros((n_x, len(alphas)))
coverage_y2 = np.zeros((n_x, len(alphas)))
coverage_both = np.zeros((n_x, len(alphas)))

data = synthetic_data.SyntheticDataWeakOverlap(rho)

for i, alpha_i in enumerate(alphas):
    # Target quantiles
    quantiles = [alpha_i, 1]
    quantiles_forest = [quantiles[0] * 100, quantiles[1] * 100]

    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        cv_random_state = seed
        params_qforest["random_state"] = cv_random_state

        # Generate synthetic data
        (z, x, y) = data.gen_synthetic_data(n_train)
        (z_test, x_test, y_test) = data.gen_synthetic_data(n_test)

        # Define the QRF model
        quantile_estimator = helper.QuantileForestRegressorAdapter(
            model=None,
            fit_params=None,
            quantiles=quantiles_forest,
            params=params_qforest)

        trainer = TrainAndEvaluate(n_x, gmm_components=2)
        weighter = BaseWeighter(trainer, delta_inf=10000)

        nc = RegressorNc(quantile_estimator,
                         WeightedQuantileRegErrFunc(weighter))

        # Run CQR procedure
        ci_lower_all, ci_upper_all, coverage_yi, coverage_both_i = helper_weighted_cqr.run_ic_per_x(
            nc,
            z,
            x,
            y,
            z_test,
            alpha_i,
            weighter,
            data,
            print_coverage=False,
            test_coverage=True)

        coverage_y1[:, i] = coverage_y1[:, i] + coverage_yi[0]
        coverage_y2[:, i] = coverage_y2[:, i] + coverage_yi[1]
        coverage_both[:, i] = coverage_both[:, i] + coverage_both_i

        mean_coverage_y1 = coverage_y1[:, i] / (seed - shift)
        mean_coverage_y2 = coverage_y2[:, i] / (seed - shift)
        mean_coverage_both = coverage_both[:, i] / (seed - shift)

        if (seed - shift) % 25 == 0:
            print('Run: {}, alpha: {}'.format(seed - shift, alpha_i))
            print('y1: {}'.format(mean_coverage_y1))
            print('y2: {}'.format(mean_coverage_y2))
            print('both: {}'.format(mean_coverage_both))

mean_y1 = coverage_y1 / n_mc
mean_y2 = coverage_y2 / n_mc
mean_both = coverage_both / n_mc

np.savez('mean_coverage',
         mean_y1=mean_y1,
         mean_y2=mean_y2,
         mean_both=mean_both,
         alphas=alphas,
         rho=rho)

"""
To replicate results in section 4.1 
in "Learning Pareto-Efficient Decisions with Confidence".

"""

import numpy as np
import synthetic_data
import pandas as pd
import random
import helper_weighted_cqr

from train_and_evaluate import TrainAndEvaluate
from cqr.cqr_weighter import BaseWeighter

from cqr import helper
from cqr.nc import RegressorNc
from cqr.nc import WeightedQuantileRegErrFunc

n_train = 1000
n_test = 100
n_x = 5

# Set seed
seed = 123
random.seed(seed)
np.random.seed(seed)

# Generate synthetic data
data = synthetic_data.SyntheticDataWeakOverlap()
(z, x, y) = data.gen_synthetic_data(n_train)
(z_test, x_test, y_test) = data.gen_synthetic_data(n_test)

d = {
    'Age': z.flatten(),
    'Drug': x.flatten(),
    'y1': y[:, 0].flatten(),
    'y2': y[:, 1].flatten()
}
df_training = pd.DataFrame(d)

# %% Conformal intervals (See QuantileForestRegressorAdapter class in helper.py)

# Parameters random forest
n_estimators = 1000  # number of trees in the forest
min_samples_leaf = 1  # minimum number of samples at a leaf node
max_features = 'auto'  # z.shape[1] # + x.shape[1]  # number of features to consider when looking for the best split

# Parameters cross-validation
cv_qforest = False  # is cross-validation used to tune the quantile levels?
cv_random_state = seed

# Define the QRF's parameters
params_qforest = dict()
params_qforest["n_estimators"] = n_estimators
params_qforest["min_samples_leaf"] = min_samples_leaf
params_qforest["max_features"] = max_features
params_qforest["CV"] = cv_qforest
params_qforest["random_state"] = cv_random_state

# Levels
alpha_i = 0.1  # Called alpha/2 in paper

# Target quantiles
quantiles = [alpha_i, 1]
quantiles_forest = [quantiles[0] * 100, quantiles[1] * 100]

# Define the QRF model
quantile_estimator = helper.QuantileForestRegressorAdapter(
    model=None,
    fit_params=None,
    quantiles=quantiles_forest,
    params=params_qforest)

trainer = TrainAndEvaluate(n_x, gmm_components=2)
weighter = BaseWeighter(trainer, delta_inf=10000)
nc = RegressorNc(quantile_estimator, WeightedQuantileRegErrFunc(weighter))

# Run CQR procedure
ci_lower_all, ci_upper_all, __, __ = helper_weighted_cqr.run_ic_per_x(
    nc, z, x, y, z_test, alpha_i, weighter, data)

# Save the results
np.savez('ci_all_{}'.format(int(alpha_i * 100)),
         ci_lower_all=ci_lower_all,
         ci_upper_all=ci_upper_all,
         z_test=z_test,
         alpha_i=alpha_i)
df_training.to_pickle('df_training_synthetic_{}.pkl'.format(int(alpha_i *
                                                                100)))

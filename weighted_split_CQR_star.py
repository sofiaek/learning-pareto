"""
To replicate results in section 4.2 
in "Learning Pareto-Efficient Decisions with Confidence".
"""

import numpy as np
import torch
import helper_weighted_cqr
import synthetic_data_star

from train_and_evaluate import TrainAndEvaluateNoShift
from cqr.cqr_weighter import BaseWeighter

from cqr import helper
from cqr.nc import RegressorNc
from cqr.nc import WeightedQuantileRegErrFunc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from preprocess_star import get_preprocessed_star_data

# used to determine the size of test set
test_ratio = 0.2
n_x = 3

seed = 10
np.random.seed(seed)
torch.manual_seed(seed)

x, y1, z_num, z_cat, z_cat_one_hot = get_preprocessed_star_data()

idx = list(range(len(x)))
train_idx, test_idx = train_test_split(idx,
                                       test_size=test_ratio,
                                       random_state=seed)

# Scale the covariates/features
z_scaler_num = StandardScaler()
z_scaler_num.fit(z_num[train_idx])
z_num_scaled = z_scaler_num.transform(z_num)
z_scaled = np.hstack((z_num_scaled, z_cat_one_hot))

z_scaler_cat = StandardScaler()
z_scaler_cat.fit(z_cat[train_idx])
z_cat_scaled = z_scaler_cat.transform(z_cat)
z_all_scaled = np.hstack((z_num_scaled, z_cat_scaled))

data = synthetic_data_star.SyntheticOutcomeStar()
y2 = data.gen_outcome(x, z_all_scaled)
y_tot = np.hstack((y1, -y2))  # y2 is cost -> -y2 reward

# Separate into test and training sets
z_train, z_test = z_scaled[train_idx], z_scaled[test_idx]
x_train, x_test = x[train_idx], x[test_idx]
y_train, y_test = y_tot[train_idx], y_tot[test_idx]

#####################################################
# Neural network parameters
# (See AllQNet_RegressorAdapter class in helper.py)
#####################################################

nn_learn_func = torch.optim.Adam
epochs = 100
lr = 0.05
batch_size = 64
hidden_size = 16
dropout = 0.0
in_shape = z_train.shape[1]

# weight decay regularization
wd = 1e-3

# ratio of held-out data, used in cross-validation
cv_test_ratio = 0.2

# seed for splitting the data in cross-validation.
cv_random_state = 1

# Levels
alpha_i = 0.1

# Target quantiles
quantiles_net = [alpha_i, 1 - alpha_i]

# define quantile neural network model
quantile_estimator = helper.AllQNet_RegressorAdapter(
    model=None,
    fit_params=None,
    in_shape=in_shape,
    hidden_size=hidden_size,
    quantiles=quantiles_net,
    learn_func=nn_learn_func,
    epochs=epochs,
    batch_size=batch_size,
    dropout=dropout,
    lr=lr,
    wd=wd,
    test_ratio=cv_test_ratio,
    random_state=cv_random_state,
    use_rearrangement=False)

trainer = TrainAndEvaluateNoShift(n_x=3)
weighter = BaseWeighter(trainer, delta_inf=10000)
nc = RegressorNc(quantile_estimator, WeightedQuantileRegErrFunc(weighter))

# Run CQR procedure
ci_lower_all, __, __, __ = helper_weighted_cqr.run_ic_per_x(
    nc,
    z_train,
    x_train,
    y_train,
    z_test,
    alpha_i,
    weighter,
    data,
    print_coverage=False,
    test_coverage=False)

ci_y2_upper = -np.array(ci_lower_all[1])
ci_y1_lower = np.array(ci_lower_all[0])

z_original = np.hstack((z_num, z_cat))

np.savez('ci_yi_{}_star'.format(int(alpha_i * 100)),
         ci_y1=ci_y1_lower,
         ci_y2=ci_y2_upper,
         dropout=dropout,
         lr=lr,
         hidden_size=hidden_size,
         batch_size=batch_size,
         wd=wd)

np.savez('data_{}_star'.format(int(alpha_i * 100)),
         x_train=x_train,
         y_train=y_train,
         z_train=z_train,
         x_test=x_test,
         y_test=y_test,
         z_test=z_test,
         z_orig=z_original)

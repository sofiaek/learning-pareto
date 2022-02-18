"""
To replicate coverage results in section 4.2 
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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedKFold
from preprocess_star import get_preprocessed_star_data

n_x = 3

alphas = (0.05, 0.10, 0.15)

coverage_y1_mean = np.zeros((len(alphas)))
coverage_y2_mean = np.zeros((len(alphas)))
coverage_both_mean = np.zeros((len(alphas)))

correct_y1 = np.zeros((n_x, len(alphas)))
correct_y2 = np.zeros((n_x, len(alphas)))
correct_y2_full = np.zeros((n_x, len(alphas)))
correct_both = np.zeros((n_x, len(alphas)))
sum_x = np.zeros((n_x, len(alphas)))

x, y1, z_num, z_cat, z_cat_one_hot = get_preprocessed_star_data()

splits = 5
repeats = 100

n = splits * repeats
rkf = RepeatedKFold(n_splits=splits, n_repeats=repeats, random_state=2652124)
for i, alpha_i in enumerate(alphas):
    seed = 0
    for train_index, test_index in rkf.split(x):
        seed += 1
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Scale the covariates/features
        z_scaler_num = StandardScaler()
        z_scaler_num.fit(z_num[train_index])
        z_num_scaled = z_scaler_num.transform(z_num)
        z_scaled = np.hstack((z_num_scaled, z_cat_one_hot))

        z_scaler_cat = StandardScaler()
        z_scaler_cat.fit(z_cat[train_index])
        z_cat_scaled = z_scaler_cat.transform(z_cat)
        z_all_scaled = np.hstack((z_num_scaled, z_cat_scaled))

        data = synthetic_data_star.SyntheticOutcomeStar()
        y2, y2_test = data.gen_outcome_test(x, z_all_scaled,
                                            z_all_scaled[test_index])
        y_tot = np.hstack((y1, -y2))  # W is cost -> -w reward

        # Separate into test and training sets
        z_train, z_test = z_scaled[train_index], z_scaled[test_index]
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y_tot[train_index], y_tot[test_index]

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
        wd = 1e-3
        in_shape = z_train.shape[1]
        cv_test_ratio = 0.2
        cv_random_state = 1
        quantiles = [alpha_i, 1 - alpha_i]
        quantiles_net = [quantiles[0], quantiles[1]]
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
        nc = RegressorNc(quantile_estimator,
                         WeightedQuantileRegErrFunc(weighter))

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

        ci_y2_upper = np.array(ci_lower_all[1])
        ci_y1_lower = np.array(ci_lower_all[0])

        # Save the result
        correct_y1_all = 0
        correct_y2_all = 0
        correct_both_all = 0

        for x_i in range(n_x):
            x_map = x_test == x_i
            x_map = x_map.reshape(-1, )

            correct_y1[x_i,
                       i] += np.sum(y_test[x_map, 0] > ci_y1_lower[x_i, x_map])
            correct_y2[x_i,
                       i] += np.sum(y_test[x_map, 1] > ci_y2_upper[x_i, x_map])
            sum_x[x_i, i] += np.sum(x_map)

            both_i = np.logical_and(y_test[x_map, 0] > ci_y1_lower[x_i, x_map],
                                    y_test[x_map, 1] > ci_y2_upper[x_i, x_map])
            correct_both[x_i, i] += np.sum(both_i)

            correct_y1_all += np.sum(
                y_test[x_map, 0] > ci_y1_lower[x_i, x_map])
            correct_y2_all += np.sum(
                y_test[x_map, 1] > ci_y2_upper[x_i, x_map])
            correct_both_all += np.sum(both_i)

            correct_y2_full[x_i, i] += np.sum(
                -y2_test[:, x_i] > ci_y2_upper[x_i, :])

        coverage_y1_mean[i] += correct_y1_all / len(x_test)
        coverage_y2_mean[i] += correct_y2_all / len(x_test)
        coverage_both_mean[i] += correct_both_all / len(x_test)

        mean_coverage_y1 = correct_y1[:, i] / sum_x[:, i]
        mean_coverage_y2 = correct_y2[:, i] / sum_x[:, i]
        mean_coverage_both = correct_both[:, i] / sum_x[:, i]

        mean_coverage_y2_full = correct_y2_full[:, i] / np.sum(sum_x[:, i])
        if seed % 10 == 0:
            print('Seed: {}, alpha: {}'.format(seed, alpha_i))
            print('y1: {}'.format(mean_coverage_y1))
            print('y2: {}'.format(mean_coverage_y2))
            print('Both: {}'.format(mean_coverage_both))
            print('y2 all: {}'.format(mean_coverage_y2_full))

coverage_y1 = correct_y1 / sum_x
coverage_y2 = correct_y2 / sum_x
coverage_both = correct_both / sum_x

coverage_y1_mean = coverage_y1_mean / n
coverage_y2_mean = coverage_y2_mean / n
coverage_both_mean = coverage_both_mean / n

coverage_y2_full = correct_y2_full / np.sum(sum_x, axis=0)

np.savez('mean_coverage_star',
         coverage_y1_per_x=coverage_y1,
         coverage_y2_per_x=coverage_y2,
         coverage_both_per_x=coverage_both,
         coverage_y1_mean=coverage_y1_mean,
         coverage_y2_mean=coverage_y2_mean,
         coverage_both_mean=coverage_both_mean,
         correct_y2_synthetic=correct_y2_full,
         coverage_y2_synthetic=coverage_y2_full,
         sum_x=sum_x,
         alphas=alphas)

"""
Helper class to run the CQR code.  

@author: Sofia Ek
"""

import numpy as np

from cqr import helper


def run_ic_per_x(nc,
                 Z_train_orig,
                 x_train_orig,
                 y_train_orig,
                 Z_test,
                 alpha,
                 weighter,
                 data_generator,
                 print_coverage=True,
                 test_coverage=True):
    n_test = Z_test.shape[0]
    n_x = weighter.get_n_x()
    n_y = y_train_orig.shape[1]

    ci_lower_list = []
    [ci_lower_list.append([0] * n_x) for y in range(n_y)]

    ci_upper_list = []
    [ci_upper_list.append([0] * n_x) for y in range(n_y)]

    weighter.train_weights(Z_train_orig, x_train_orig)

    coverage_y = []
    [coverage_y.append([]) for y in range(n_y)]
    coverage_both = []

    for x in range(n_x):
        x_map = x_train_orig == x
        x_map = x_map.reshape(-1, )

        x_test = x * np.ones((n_test, 1), dtype=int)
        Z_train = Z_train_orig[x_map].reshape(-1, Z_train_orig.shape[1])
        x_train = x_train_orig[x_map].reshape(-1, 1)
        y_train = y_train_orig[x_map].reshape(-1, y_train_orig.shape[1])

        n_train = Z_train.shape[0]

        # divide the data into proper training set and calibration set
        idx = np.random.permutation(n_train)
        n_half = int(np.floor(n_train / 2))
        idx_train, idx_cal = idx[:n_half + 1], idx[n_half + 1:2 * n_half]

        p = np.zeros((n_x))
        p[x] = 1

        weighter.set_data(Z_train[idx_cal, :],
                          x_train[idx_cal],
                          Z_test,
                          x_test,
                          prob=p)

        for i in range(y_train_orig.shape[1]):
            ci_lower_list[i][x], ci_upper_list[i][x] = helper.run_icp(
                nc, Z_train, y_train[:, i].reshape(-1), Z_test, idx_train,
                idx_cal, alpha)

        if test_coverage:
            # Compute (and print) average coverage
            y_test = data_generator.gen_outcome(Z_test, x_test)

            if print_coverage:
                print('Testing coverage, x = {}'.format(x))

            for i in range(n_y):
                coverage = helper.compute_one_sided_coverage(
                    y_test[:, i], ci_lower_list[i][x], alpha, "CQR")
                coverage_y[i].append(coverage)
                if print_coverage:
                    print(
                        "CQR for y{}: Percentage in the range (expecting {}): {}"
                        .format(i + 1, 100 - alpha * 100, coverage))

            if n_y == 2:
                coverage = helper.compute_coverage_one_sided_2dim(
                    y_test, ci_lower_list[0][x], ci_lower_list[1][x],
                    alpha * 2, "CQR for y1 and y2")
                coverage_both += [coverage]
                if print_coverage:
                    print(
                        "CQR for y1 and y2: Percentage in the range (expecting {}): {}"
                        .format(100 - alpha * 2 * 100, coverage))

    return ci_lower_list, ci_upper_list, coverage_y, coverage_both

"""
Weights the probabilies in the CQR algorithm in case of a covariate shift

@author: Sofia Ek
"""

import numpy as np


class BaseWeighter:
    def __init__(self, trainer, delta_inf=10000):
        self.z_cal, self.x_cal = 0, 0
        self.z_test, self.x_test = 0, 0
        self.prob = 1
        self.trainer = trainer
        self.delta_inf = delta_inf

    def train_weights(self, z_train, x_train):
        self.trainer.train_prob(x_train, z_train)

    def get_n_x(self):
        return self.trainer.get_n_x()

    def set_data(self, z_cal, x_cal, z_test, x_test, prob):
        self.z_cal = z_cal
        self.z_test = z_test
        self.x_cal = x_cal
        self.x_test = x_test
        self.prob = prob

    # Bullet 3
    def _weights_x(self, z, x, train):
        num = self.trainer.get_num_weight_x(z) * self.trainer.get_p_x(
            self.prob, x)
        den = self.trainer.get_den_weight_x(x, z)
        wt = num / den
        return wt

    # Bullet 4
    def _get_normalized_weights(self, wt_trn, wt_test):
        if wt_test.shape[0] > 1:
            print('Only one test point supported.')
            return
        p_k_train = wt_trn / (np.sum(wt_trn) + wt_test)
        p_k_test = wt_test / (np.sum(wt_trn) + wt_test)
        return p_k_train, p_k_test

    # Testing
    def _get_equal_weights(self, wt_trn, wt_test):
        if wt_test.shape[0] > 1:
            print('Only one test point supported.')
            return
        test = np.ones((wt_trn.shape[0]))  # No shift
        p_k_train = test / (np.sum(test) + 1)  # No shift
        p_k_test = 1 / (np.sum(test) + 1)  # No shift
        return p_k_train, p_k_test

    # Bullet 5
    def get_eta(self, nc, alpha):
        n_test = self.x_test.shape[0]
        wt_trn = self._weights_x(self.z_cal, self.x_cal, True)
        wt_test = self._weights_x(self.z_test, self.x_test, False)
        eta = np.zeros(n_test)
        for i in range(n_test):
            p_k_train, p_k_test = self._get_normalized_weights(
                wt_trn, wt_test[i])
            eta[i] = self._generate_quantile_score(p_k_train, p_k_test, nc,
                                                   alpha)
        return eta

    def _generate_quantile_score(self, p_x_train, p_x_test, nc, alpha):
        # p_x_train and s_i are vectors for the whole data set evaluated in the current test point
        # p_x_test and s are scalars for the current test point
        p_x = np.append(p_x_train, p_x_test)
        scr = np.append(nc, self.delta_inf)
        srt_idx = np.argsort(scr)
        srt_prb = p_x[srt_idx]
        srt_scr = scr[srt_idx]
        F = 0
        i = 0
        n1 = np.size(scr)
        while i in range(n1):
            F += srt_prb[i]
            if F >= (1 - alpha):
                break
            else:
                i += 1
        if i == n1:
            return srt_scr[i - 1]
        else:
            return srt_scr[i]

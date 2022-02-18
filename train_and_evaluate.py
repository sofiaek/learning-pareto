"""
Trains and evaluates the probabilites of the weights.

z = Continuous
x = Discrete

@author: Sofia Ek
"""

import numpy as np
from sklearn import mixture


class TrainAndEvaluateNoShift:
    "Base trainer. Assumes no covariate shifts."

    def __init__(self, n_x):
        self.n_x = n_x  # Number of x treatments
        self.p_x = np.zeros((n_x))

    def get_n_x(self):
        return self.n_x

    def train_prob(self, x, z):
        self._train_p_x(x)

    def _train_p_x(self, x):
        for x_i in range(self.n_x):
            self.p_x[x_i] = np.sum(x == x_i) / np.size(x)

    def get_num_weight_x(self, z):
        return 1

    def get_den_weight_x(self, x, z):
        return self._evaluate_p_x(x)

    def get_p_x(self, p_x_test, x):
        p = p_x_test[np.array(x, dtype=int)]
        return p

    def _evaluate_p_x(self, x):
        p = self.p_x[np.array(x, dtype=int)]
        return p


class TrainAndEvaluate:
    "Trains and evaluates the probabilites for a one dimensional z."

    def __init__(self, n_x, gmm_components):
        self.n_x = n_x  # Number of x treatments
        self.gmm_components = gmm_components

        self.p_x = np.zeros((n_x))
        self.gm_z_x = []  # p(z_age | x)

    def get_n_x(self):
        return self.n_x

    def train_prob(self, x, z):
        self.gm_z_x = []

        # Marginal/joint probabilities
        self._train_p_x(x)
        # Conditional probabilitites
        self._train_p_z_x(z, x)

    def _train_p_x(self, x):
        for x_i in range(self.n_x):
            self.p_x[x_i] = np.sum(x == x_i) / np.size(x)

    def _train_p_z_x(self, z, x):
        for x_i in range(self.n_x):
            gmm = mixture.GaussianMixture(n_components=self.gmm_components,
                                          covariance_type='full')
            gmm.fit(z[x == x_i].reshape(-1, 1))

            self.gm_z_x.append(gmm)

    def get_num_weight_x(self, z):
        p_z = self._evaluate_p_z(z)
        return p_z

    def get_den_weight_x(self, x, z):
        p = self._evaluate_p_x(x) * self._evaluate_p_z_x(z, x)
        return p

    def get_p_x(self, p_x_test, x):
        p = p_x_test[np.array(x, dtype=int)]
        return p

    def _evaluate_p_z(self, z):
        x_test = np.ones((z.shape[0], 1))
        pz = 0
        for x_i in range(self.n_x):
            pz = pz + self._evaluate_p_z_x(
                z, x_i * x_test) * self._evaluate_p_x(x_i)
        return pz

    def _evaluate_p_z_x(self, z, x):
        n = np.size(x)
        p = np.zeros((n, 1))
        for i in range(n):
            data = z[i].reshape(-1, 1)

            p[i] = np.exp(self.gm_z_x[int(x[i])].score_samples(data))
        return p

    def _evaluate_p_x(self, x):
        p = self.p_x[np.array(x, dtype=int)]
        return p

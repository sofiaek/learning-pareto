"""
To create the synthetic data used as cost in section 4.2.

@author: Sofia Ek
"""

import numpy as np


class SyntheticOutcomeStar:
    def __init__(self):
        self.sigma_y = 1
        self.tau0 = 4
        self.tau2 = 2
        self.mu_k = 10

        #weights used in IHDP data generator
        self.val_beta = np.array([0, 0.1, 0.2, 0.3, 0.4])
        #probabilities to sample beta with
        self.prob_beta = np.array([0.4, 0.15, 0.15, 0.15, 0.15])

    def draw_beta(self, d):
        beta = np.random.choice(self.val_beta, p=self.prob_beta, size=(d, 1))
        return beta

    def gen_outcome(self, x, z):
        d = np.shape(z)[1]
        beta = self.draw_beta(d)
        # z = z.reshape((1, -1))
        mu1 = np.exp((z + 0.5).dot(beta)) + self.mu_k
        mu_temp = z.dot(beta) + self.mu_k

        omega0 = np.mean(mu_temp[x == 0] - mu1[x == 0]) - self.tau0
        omega2 = np.mean(mu_temp[x == 2] - mu1[x == 2]) - self.tau2

        mu0 = mu_temp - omega0
        mu2 = mu_temp - omega2

        mean = np.zeros((len(x), 1))
        mean[x == 0] = mu0[x == 0]
        mean[x == 1] = mu1[x == 1]
        mean[x == 2] = mu2[x == 2]

        y = np.random.normal(mean, self.sigma_y)

        return y

    def gen_outcome_test(self, x, z, z_test):
        d = np.shape(z)[1]
        beta = self.draw_beta(d)

        mu1 = np.exp((z + 0.5).dot(beta)) + self.mu_k
        mu_temp = z.dot(beta) + self.mu_k

        omega0 = np.mean(mu_temp[x == 0] - mu1[x == 0]) - self.tau0
        omega2 = np.mean(mu_temp[x == 2] - mu1[x == 2]) - self.tau2

        mu0 = mu_temp - omega0
        mu2 = mu_temp - omega2

        mean = np.zeros((len(x), 1))
        mean[x == 0] = mu0[x == 0]
        mean[x == 1] = mu1[x == 1]
        mean[x == 2] = mu2[x == 2]

        y = np.random.normal(mean, self.sigma_y)

        # Test data
        mu1_test = np.exp((z_test + 0.5).dot(beta)) + self.mu_k
        mu_temp = z_test.dot(beta) + self.mu_k
        mu0_test = mu_temp - omega0
        mu2_test = mu_temp - omega2

        y_test = np.zeros((z_test.shape[0], 3))
        y_test[:, 0] = np.random.normal(mu0_test, self.sigma_y).flatten()
        y_test[:, 1] = np.random.normal(mu1_test, self.sigma_y).flatten()
        y_test[:, 2] = np.random.normal(mu2_test, self.sigma_y).flatten()

        return y, y_test

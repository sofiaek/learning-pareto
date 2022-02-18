"""
To create the synthetic data used in the Numerical Experiments in section 4.1.  

@author: Sofia Ek
"""

import numpy as np


class SyntheticDataBase:
    """Base class to generate the synthetic data with uniform random assignments. 
    """
    def __init__(self, rho=-0.2):
        # parameters for the covariate distribution
        # covariate = [z]
        (self.mu, self.s) = (60, 10)

        # Parameters to generate outcome (y1, y2)
        # y1 = a_x + b_x * z + u_y1
        # y2 = c_x + d_x * z + u_y2
        self.a_x = np.array([0, 2.2, 0.6, 0.0, 2.2])
        self.b_x = np.array([2.4, -1.5, 1, 2, -1])
        self.c_x = np.array([2.4, 0.7, 0.8, 2.0, 1.2])
        self.d_x = np.array([-1.4, 1.5, 1, -1.2, 1])

        # variance of outcome
        self.sigma = 0.2
        self.rho = rho

    def gen_synthetic_data(self, n):
        # covariates
        z = self._gen_covariates(n)
        # decision
        x = self._gen_decision(z)
        # outcome (y1 and y2)
        y = self.gen_outcome(z, x)
        return z, x, y

    def gen_outcome(self, z, x):
        # generate outcome
        n = np.size(x)
        mean = np.array([0, 0])
        cov = self.sigma**2 * np.array(([1, self.rho], [self.rho, 1]))
        noise = np.random.multivariate_normal(mean, cov, n)
        y1 = self.a_x[x] + self._get_b(x, z) + noise[:, 0].reshape(-1, 1)
        y2 = self.c_x[x] + self._get_d(x, z) + noise[:, 1].reshape(-1, 1)
        y = np.append(y1, y2, axis=1)
        return y

    def _gen_covariates(self, n):
        # generate  n x 1 array of covariates z
        z = np.random.normal(self.mu, self.s, size=n).reshape(-1, 1)
        return z

    def _gen_decision(self, z):
        # generate n x 1 array of decisions x
        # z is n x 1 array of covariates
        n = z.shape[0]
        p_x = [0.2, 0.2, 0.2, 0.2, 0.2]
        x = np.random.choice(5, size=n, p=p_x).reshape(-1, 1)
        return x

    def _get_b(self, x, z):
        return self.b_x[x] * self._sigmoid((z - 50) / 8)

    def _get_d(self, x, z):
        return self.d_x[x] * self._sigmoid((z - 55) / 9)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


class SyntheticDataWeakOverlap(SyntheticDataBase):
    """To generate the synthetic data with weak covariate overlap. 
    """
    def __init__(self, rho=-0.2):
        super().__init__(rho)
        # parameters for distribution decisions
        (self.p0, self.p1, self.p2, self.p3) = (0.2, 0.2, 0.2, 0.2)

    def gen_synthetic_data(self, n):
        # covariates
        z = super()._gen_covariates(n)
        # decision
        x = self._gen_decision(z)
        # outcome (y1 and y2)
        y = super().gen_outcome(z, x)
        return z, x, y

    def _gen_decision(self, z):
        # generate n x 1 array of decisions
        # z is n x 1 array of covariates
        n = z.shape[0]
        p_x = np.random.uniform(size=(n, 1))
        p_age = self._sigmoid((-z + self.mu + 10) / 5)
        p_x = p_x * p_age
        x = np.zeros((n, 1), dtype=int)
        for i in range(n):
            if p_x[i] < self.p0:
                x[i] = 0
            elif p_x[i] < self.p0 + self.p1:
                x[i] = 1
            elif p_x[i] < self.p0 + self.p1 + self.p2:
                x[i] = 2
            elif p_x[i] < self.p0 + self.p1 + self.p2 + self.p3:
                x[i] = 3
            else:
                x[i] = 4
        return x

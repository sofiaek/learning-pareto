"""
To find the Pareto-efficient decisions.
"""

import numpy as np


def is_pareto_front(y1, y2):
    """To find the pareto-front.

	Parameters
	----------
	y1 : numpy array of shape [n_decisions, 1]
		One of the rewards

	y2 : numpy array of shape [n_decisions, 1]
		The other reward

	Returns
		-------
	is_front : Numpy boolean array of shape [n_decisions, ]
               Indicates whether a point is on the Pareto front
	"""
    is_front = np.ones(y1.shape[0], dtype=bool)
    reward = np.append(y1, y2, axis=1).reshape(-1, 2)
    for i, r in enumerate(reward):
        if is_front[i]:
            is_front[is_front] = np.any(reward[is_front] > r, axis=1)
            is_front[i] = True
    return is_front


def alpha_pareto(y1, y2):
    """To find the alpha-pareto front.

	Parameters
	----------
	y1 : numpy array of shape [n_samples, n_decisions = 2]
		One of the rewards

	y2 : numpy array of shape [n_samples, n_decisions = 2]
		The other reward

	Returns
		-------
	y1_mean_alpha : Numpy array of shape [n_alpha = 10, ]
               One of the mean rewards on the alpha-Pareto front
               
    y2_mean_alpha : Numpy array of shape [n_alpha = 10, ]
               The other mean reward on the alpha-Pareto front
            
	"""
    y1_diff = y1[:, 1] - y1[:, 0]
    y2_diff = y2[:, 1] - y2[:, 0]

    n_alpha = 10
    alpha_pareto = np.linspace(0.0, 1.0, num=n_alpha)
    y1_mean_alpha = np.zeros((n_alpha))
    y2_mean_alpha = np.zeros((n_alpha))

    n_test = y1.shape[0]

    for i in range(n_alpha):
        x = np.zeros((n_test))
        x[y1_diff * alpha_pareto[i] + y2_diff * (1 - alpha_pareto[i]) >= 0] = 1

        y1_result = np.append(y1[x == 0, 0], y1[x == 1, 1])
        y1_mean_alpha[i] = np.mean(y1_result)

        y2_result = np.append(y2[x == 0, 0], y2[x == 1, 1])
        y2_mean_alpha[i] = np.mean(y2_result)

    return y1_mean_alpha, y2_mean_alpha

"""
Preprocessing star

@author: Sofia Ek
"""

import numpy as np
import pandas as pd


def _to_categorical(X):
    assert (type(X) == type(np.ones((1, ))))
    return pd.get_dummies(pd.DataFrame(X.astype(str)),
                          dummy_na=True).values.astype(np.int)


def get_preprocessed_star_data():
    star_data = pd.read_csv('STAR_Students.csv', sep=';')

    # To remove students that were not part of the study in first grade, or that are missing a test score
    dec_filter = np.isfinite(star_data.g1classtype)
    out_filter = np.isfinite(star_data.g1tlistss + star_data.g1treadss +
                             star_data.g1tmathss)
    tot_filter = np.logical_and(dec_filter, out_filter)

    # Decision variable x
    x = star_data.g1classtype[tot_filter].values.reshape(-1, 1)
    x = x - 1

    # Outcome y1, an achievement test score (sum of math, reading and listening scores)
    outcome = ['g1treadss', 'g1tmathss', 'g1tlistss']
    y = star_data[outcome][tot_filter].values
    y = np.sum(y, axis=1).reshape(-1, 1)

    # Covariates/features z (numerical and categorical)
    # Numerical: missing values are set to mean value
    cov_num = ['g1present', 'g1absent', 'birthmonth', 'g1tcareer', 'g1tyears']
    z_num = star_data[cov_num][tot_filter]
    z_mean = np.mean(z_num, axis=0)
    values = {
        'g1present': z_mean[0],
        'g1absent': z_mean[1],
        'birthmonth': z_mean[2],
        'g1tcareer': z_mean[3],
        'g1tyears': z_mean[4],
    }
    z_num = star_data[cov_num][tot_filter].fillna(value=values).values
    # Categorical: Missing values are set to 0.
    cov_cat = [
        'gender', 'race', 'g1surban', 'g1promote', 'g1specin', 'g1freelunch'
    ]

    z_cat = star_data[cov_cat][tot_filter].fillna(0).values
    z_cat_one_hot = _to_categorical(
        star_data[cov_cat][tot_filter].fillna(0).values)

    return x, y, z_num, z_cat, z_cat_one_hot

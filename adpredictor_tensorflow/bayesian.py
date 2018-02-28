# coding: utf-8
import numpy as np
import tensorflow as tf
from scipy.stats import norm

import utils


def update(dists, X, y, beta, epsilon):
    total_mean, total_var = active_mean_variance(dists, X, beta)
    v, w = utils.gaussian_corrections(y * total_mean / np.sqrt(total_var))

    for index, feature in enumerate(X):
        mean, var = dists[index][feature]
        mean_delta = y * var / np.sqrt(total_var) * v
        var_coeff = 1. - var / total_var * w
        dists[index][feature] = apply_dynamics(mean + mean_delta, var * var_coeff, epsilon)

    return dists

def active_mean_variance(dists, X, beta):
    pairs = [ dists[i][f] for i, f in enumerate(X) ]
    mean = sum([ p[0] for p in pairs ])
    var = sum([ p[1] for p in pairs ]) + beta ** 2
    return mean, var

def apply_dynamics(mean, var, epsilon):
    prior_mean, prior_var = utils.get_prior_params()
    new_var = var * prior_var / ((1. - epsilon) * prior_var + epsilon * var)
    new_mean = new_var * ((1. - epsilon) * mean / var + epsilon * prior_mean / prior_var)
    return [ new_var, new_mean ]

def predict(dists, X, beta):
    total_mean, total_var = active_mean_variance(dists, X, beta)
    return norm.cdf(total_mean / total_var)

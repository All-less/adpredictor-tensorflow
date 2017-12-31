# coding: utf-8
from scipy.stats import norm
import numpy as np


def get_dists_init(config):
    """Generate initial values of all normal distributions.
    """
    init = []

    # Each possible value of each feature has a normal distribution associated to it.
    for f in range(config.num_features):
        feature_init = []
        for v in range(config.feature_max + 1):  # we spare one cell for bias
            feature_init.append([ 0., 1.])  # (mean, variance)
        init.append(feature_init)

    # We adjust the mean of a bias so that, with the given parameters,
    # P(y | x, initial_weights) = prior.
    bias_mean = norm.ppf(config.prior_prob) * (config.beta ** 2 + config.num_features)
    init[0][0] = [ bias_mean, 1. ]
    return init


MAX_ABS_SURPRISE = 5.0

def gaussian_corrections(t):
    """Returns the additive and multiplicative corrections for the mean
    and variance of a trunctated Gaussian random variable.

    In Trueskill/AdPredictor papers, denoted
    - V(t)
    - W(t) = V(t) * (V(t) + t)

    Returns (v(t), w(t))
    """
    # Clipping avoids numerical issues from ~0/~0.
    t = np.clip(t, -MAX_ABS_SURPRISE, MAX_ABS_SURPRISE)
    v = norm.pdf(t) / norm.cdf(t)
    w = v * (v + t)
    return v, w

def get_prior_params():
    """The global prior on non-bias weights
    """
    return 0., 1.

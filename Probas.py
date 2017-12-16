import numpy as np


def norm_exp_logweights(lw):
    """
    Compute a quantity propotionnal to the weights
    up to the multiplicative constant exp(max(lw))
    which does not matter since it simplifies out
    in the normalization, but it limits rounding errors
    which may occurs since the weights are small quantities
    :param lw: np.array, array of logweights
    :return:
    """
    w = np.exp(lw-np.max(lw))
    return w/np.sum(w)
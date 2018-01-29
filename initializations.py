import numpy as np


def xmat_inbounds(nsamples, bounds):
    k = len(bounds)
    xmat = np.zeros((nsamples, k))
    for i in range(0, k):
        xmat[:, i] = np.random.uniform(bounds[i][0], bounds[i][1], nsamples)
    return xmat


def init_y(xmat, objective_func):
    nsamples = xmat.shape[0]
    y = np.zeros((nsamples, 1))
    for i in range(0, nsamples):
        y[i, 0] = objective_func(xmat[i, :])
    return y


def xinit_inbounds(bounds):
    """
    Random initialization of xinit within given bounds

    Args :
        bounds (tuple) : bounds, for instance in 2d : ((min_d1, min_d2), (max_d1, max_d2))

    Returns:
        numpy.ndarray. The random point for initialization within bounds
    """
    k = len(bounds)
    xinit = np.zeros((k, ))
    for b in range(0, k):
        xinit[b] = np.random.uniform(bounds[b][0], bounds[b][1])
    return xinit

import numpy as np


def kernel_func(x1, x2, theta, p):
    """
    2d version of correlation kernel using product rule

    Args:
        x1 (numpy.ndarray) : first datapoint, shape = (k, )
        x2(numpy.ndarray) : second datapoint, shape = (k, )
        theta (numpy.ndarray) : vector of theta params, one by dim, shape = (k, )
        p (numpy.ndarray) : powers used to compute the distance, one by dim, shape = (k, )

    Returns:
        float. The 2d kernel correlation between xvec1 and xvec2
    """
    ndims = theta.shape[0]
    corrprod = 1.0
    for i in range(0, ndims):
        pdist = np.power(np.absolute(x1[i] - x2[i]), p[i])
        corrprod *= np.exp(-(1.0 / np.absolute(theta[i])) * pdist)
    return corrprod


def kernel_mat(xmat, theta, p):
    """
    Compute kernel matrix from a set of datapoints

    Args:
        xmat (numpy.ndarray) : the data points, shape = (n, k),
        theta (numpy.ndarray) : vector of theta params, one by dim, shape = (k, )
        p (numpy.ndarray) : powers used to compute the distance, one by dim, shape = (k, )

    Returns:
        numpy.ndarray. Kernel matrix for points in xmat
    """
    n = xmat.shape[0]
    R = np.zeros((n, n))
# We use the symmetric structure to divide
# the number of calls to corr_func_2d by 2
    for j in range(0, n):
        for i in range(j, n):
            corr = kernel_func(xmat[i, :], xmat[j, :], theta, p)
            R[i, j] = corr
            R[j, i] = corr
    return R


def kernel_rx(xmat, xnew, theta, p):
    """
    Compute rx correlation vector for new data point using classic kernel approach

    Args:
        xmat (numpy.ndarray) : the data points, shape = (n, k),
        xnew (numpy.ndarray) : the new data point, shape = (k, )
        theta (numpy.ndarray) : vector of theta params, one by dim, shape = (k, )
        p (numpy.ndarray) : powers used to compute the distance, one by dim, shape = (k, )

    Returns:
        numpy.ndarray. The kernel vector of xnew against points in xmats
    """

    n = xmat.shape[0]
    rx = np.zeros((n, 1))
    for i in range(0, n):
        rx[i, 0] = kernel_func(xmat[i, :], xnew, theta, p)
    return rx

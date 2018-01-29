import numpy as np


def cholesky_inv(a):
    """
    Compute the inverse of a symmetric matrix using Cholesky decomposition
    Args :
        a (numpy.ndarray) : the symmetric matrix to invert
    Returns :
        numpy.ndarray. The inverse of matrix a
    """
    l = np.linalg.cholesky(a)
    linv = np.linalg.inv(l)
    return np.dot(linv.T, linv)

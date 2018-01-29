import math
import numpy as np


def sin_1d(xvec):
    return math.sin(xvec[0])


def test_1d(xvec):
    if isinstance(xvec, np.ndarray):
        return np.abs(xvec[0] - 5) * np.cos(2 * xvec[0])
    else:
        return np.abs(xvec - 5) * np.cos(2 * xvec)


def test_1d_bis(xvec):
    if isinstance(xvec, np.ndarray):
        return -np.log(np.abs(xvec[0] - 5)) * np.cos(2 * xvec[0])
    else:
        return -np.log(np.abs(xvec - 5)) * np.cos(2 * xvec)


def mystery(x1, x2):
    """
    Test function for optimization

    Args:
        x1 (float) : 0 <= x1 <= 5
        x2 (float) : 0 <= x2 <= 5

    Returns:
        float. mystery_function(x1, x2)
    """

    a = 0.01 * (x2 - x1 * x1) * (x2 - x1 * x1)
    b = 2 * (2 - x2) * (2 - x2)
    c = 7 * math.sin(0.5 * x1) * math.sin(0.7 * x1 * x2)
    d = (1 - x1)**2

    return 2 + a + b + c + d


def mystery_vec(xvec):
    """
    mystery function taking vector as input

    Args:
        xvec (numpy.ndarray) : shape = (2, )
    """
    return mystery(xvec[0], xvec[1])


funcs_dic = dict()
funcs_dic["Mystery"] = mystery_vec

# The global solution has a value of -1.4565 at x = [2.5044,2.5778]
#x = [2.5044,2.5778]
# mystery_vec(x)

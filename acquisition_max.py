import numpy as np
import scipy.optimize as optimize

import exp_kernel
import prediction_formulae as pred


def complete_acq_func(
        xmat,
        xnew,
        y,
        Rinv,
        beta_hat,
        theta,
        p,
        acq_func):
    """
    Generate acquisition function for optimization with possibility
    to change easily of acquisition function

    Args:
        xmat (numpy.ndarray) : the data points, shape = (n, k)
        xnew (numpy.ndarray) : the new data point, shape = (k, )
        y (numpy.ndarray) : y, shape=(n, 1)
        Rinv (numpy.ndarray) : Inverse of R, shape=(n, n)
        beta_hat(float) : estimation of beta on the data of xmat
        theta (numpy.ndarray) : vector of theta params, one by dim, shape = (k, )
        p (numpy.ndarray) : powers used to compute the distance, one by dim, shape = (k, )
        acq_func : Instance of one of the classes in Acquisition_Functions.py file

    Returns:
        scipy.optimize.optimize.OptimizeResult. The result of optimization
    """
    rx = exp_kernel.kernel_rx(xmat, xnew, theta, p)
    hat_y = pred.y_est(rx, y, Rinv, beta_hat)
    hat_sigma = np.power(pred.sigma_sqr_est(y, rx, Rinv, beta_hat), 0.5)
    # print("xnew")
    # print(xnew)
    if acq_func.name == "EI":
        fmin = np.min(y)
        acq_func.set_fmin(fmin)
    return acq_func.evaluate(hat_y, hat_sigma)


def constraints_bounded(bounds):
    dim = len(bounds)
    constraints = []
    for i in range(0,dim):
        constraints += [{'type': 'ineq', 'fun': lambda x: x-bounds[i][0]},
                    {'type': 'ineq', 'fun': lambda x: -x+bounds[i][1]}]
    return(constraints)

def opti_acq_func(
        xmat,
        y,
        Rinv,
        beta_hat,
        theta,
        p,
        xinit,
        acq_func,
        bounds=None):
    """
    Optimize acquisition function

    Args:
        xmat (numpy.ndarray) : the data points, shape = (n, k)
        y (numpy.ndarray) : y, shape=(n, 1)
        Rinv (numpy.ndarray) : Inverse of R, shape=(n, n)
        beta_hat(float) : estimation of beta on the data of xmat
        theta (numpy.ndarray) : vector of theta params, one by dim, shape = (k, )
        p (numpy.ndarray) : powers used to compute the distance, one by dim, shape = (k, )
        xinit (numupy.ndarray) : shape=(k, ), where to start optimization
        acq_func : Instance of one of the classes in Acquisition_Functions.py file

    Returns:

    """
    constraints = constraints_bounded(bounds)
    def to_optimize(xnew):
        if acq_func.opti_way == "max":
            opti_sign = -1
        else:
            opti_sign = 1
        return float(opti_sign * complete_acq_func(xmat, xnew, y, Rinv, beta_hat, theta, p, acq_func))
    opti = optimize.minimize(
        fun=to_optimize,
        x0=xinit,
        constraints=constraints,
        method='COBYLA') #'SLSQP' originally
    return opti

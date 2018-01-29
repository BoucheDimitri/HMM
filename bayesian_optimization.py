import numpy as np
import matplotlib.pyplot as plt

import cho_inv
import visualization as viz
import prediction_formulae as pred
import exp_kernel
import acquisition_max as am
import initializations as initial


def bayesian_search(xmat,
                    y,
                    theta,
                    p,
                    xinit,
                    acq_func,
                    bounds=None,
                    constraints=None):
    """
    Search best point to sample by maximizing acquisition function

    Args:
        xmat (numpy.ndarray) : the data points so far, shape = (n, k)
        y (numpy.ndarray) : y, shape=(n, 1)
        theta (numpy.ndarray) : vector of theta params, one by dim, shape = (k, )
        p (numpy.ndarray) : powers used to compute the distance, one by dim, shape = (k, )
        xinit (numpy.ndarray) : initial value for acquisition maximization, shape = (k, )
        acq_func : Instance of one of the classes in Acquisition_Functions.py file
        bounds (tuple) : bounds for acquisition maximization in scipy

    Returns:
        numpy.ndarray. The new point to sample from given the data so far
    """
    R = exp_kernel.kernel_mat(xmat, theta, p)
    Rinv = cho_inv.cholesky_inv(R)
    beta_hat = pred.beta_est(y, Rinv)
    opti_result = am.opti_acq_func(xmat,
                                   y,
                                   Rinv,
                                   beta_hat,
                                   theta,
                                   p,
                                   xinit,
                                   acq_func,
                                   bounds)
    return opti_result


def evaluate_add(xmat, xnew, y, objective_func):
    """
    Once the point to sample from is chosen, this function evaluate
    the objective function at that point and update xmat and y to incorporate the new point

    Args:
        xmat (numpy.ndarray) : the data points so far, shape = (n, k)
        y (numpy.ndarray) : y, shape=(n, 1)
        objective_func (function) : function to minimize

    Returns:
         tuple. xmat expanded, y expanded
    """
    ynew = np.array([objective_func(xnew)])
    ynew = ynew.reshape((1, 1))
    y = np.concatenate((y, ynew))
    k = xmat.shape[1]
    xnew = xnew.reshape(1, k)
    xmat = np.concatenate((xmat, xnew))
    return xmat, y


def bayesian_opti(xmat,
                  y,
                  n_it,
                  theta,
                  p,
                  acq_func,
                  objective_func,
                  bounds=None):
    """
    Perform iterations of bayesian optimization

    Args:
        xmat (numpy.ndarray) : the data points so far, shape = (n, k)
        y (numpy.ndarray) : y, shape=(n, 1)
        theta (numpy.ndarray) : vector of theta params, one by dim, shape = (k, )
        p (numpy.ndarray) : powers used to compute the distance, one by dim, shape = (k, )
        acq_func : Instance of one of the classes in Acquisition_Functions.py file
        objective_func (function) : the objective function
        bounds (tuple) : bounds for acquisition maximization in scipy

    Returns:
        numpy.ndarray. The new point to sample from given the data so far
    """
    k = xmat.shape[1]
    for i in range(0, n_it):
        if bounds:
            xinit = initial.xinit_inbounds(bounds)
        else:
            xinit = np.random.rand(k)
        opti_result = bayesian_search(xmat,
                               y,
                               theta,
                               p,
                               xinit,
                               acq_func,
                               bounds)
        xnew = opti_result.x
        xmat, y = evaluate_add(xmat, xnew, y, objective_func)
    return xmat, y


def bayesian_opti_plot_1d(xmat,
                          y,
                          n_it,
                          theta,
                          p,
                          acq_func,
                          objective_func,
                          bounds=None):
    for i in range(0, n_it):
        print(i)
        xinit = initial.xinit_inbounds(bounds)
        if acq_func.name == "EI":
            acq_func.set_fmin(np.min(y))
        opti_result = bayesian_search(xmat,
                               y,
                               theta,
                               p,
                               xinit,
                               acq_func,
                               bounds)
        xnew = opti_result.x
        R = exp_kernel.kernel_mat(xmat, theta, p)
        Rinv = cho_inv.cholesky_inv(R)
        beta_hat = pred.beta_est(y, Rinv)
        axes = viz.bayes_opti_plot_1d(xmat,
                                      y,
                                      Rinv,
                                      beta_hat,
                                      theta,
                                      p,
                                      bounds[0],
                                      grid_size=1000,
                                      acq_func=acq_func,
                                      objective_func=objective_func)
        y_acq = - opti_result.fun
        axes[1].vlines(xnew[0], 0, y_acq, linestyles='dashed', colors='r', linewidth=2)
        plt.show()
        xmat, y = evaluate_add(xmat, xnew, y, objective_func)

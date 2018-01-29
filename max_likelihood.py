import math
import numpy as np
import scipy.optimize as optimize

import prediction_formulae as pred
import cho_inv
import exp_kernel


def params_to_vec(params_vec):
    """
    Separate parameters stacked together
    We stack them together to be able to use scipy optimization functions

    Args :
        params_vec (numpy.ndarray)

    Returns :
         tuple. (theta_vec, p_vec)
    """
    dim = int(params_vec.shape[0] / 2)
    theta_vec = params_vec[0: dim]
    p_vec = params_vec[dim:]
    return theta_vec, p_vec


def hat_sigmaz_sqr_mle(y, Rinv):
    """
        hat_sigmaz depending only on R and and doing the computations
        for beta. Will be useful for Mle estimation

    Args :
        y (numpy.ndarray) : shape = (n, 1)
        R (numpy.ndarray) : Kernel matrix

        Returns :
        float. estimation of sigmaz_sqr
    """
    # Rinv = cho_inv.cholesky_inv(R)
    hat_beta = pred.beta_est(y, Rinv)
    return pred.hat_sigmaz_sqr(y, Rinv, hat_beta)


def log_likelihood(xmat, y, params_vec):
    """
    Log likelihood

    Args :
        xmat (numpy.ndarray) : shape = (n, k)
        y (numpy.ndarray) : shape = (n, 1)
        params_vec (numpy.ndarray) : shape = (2*k, )

        Returns :
        float. log likelihood
    """
    theta_vec, p_vec = params_to_vec(params_vec)
    R = exp_kernel.kernel_mat(xmat, theta_vec, p_vec)
    n = R.shape[0]
    Rinv = cho_inv.cholesky_inv(R)
    detR = np.linalg.det(R)
    hat_sigz_sqr = hat_sigmaz_sqr_mle(y, Rinv)
    # print("Theta vec" + str(theta_vec))
    # print("p_vec" + str(p_vec))
    # print("sigma " + str(hat_sigz_sqr))
    # print("Det " + str(detR))
    return - 0.5 * (n * math.log(hat_sigz_sqr) + math.log(detR))


def bounds_generator(mins_list, maxs_list):
    """
    Generate bounds to constrain value range in likelihood maximization

    Args:
        mins_list (list) : list of lower bounds
        maxs_list (list) : list of upper bounds

    Returns:
        list. A list of bounds that is scipy.optimize friendly
    """
    bounds = []
    for i in range(0, len(mins_list)):
        bounds.append((mins_list[i], maxs_list[i]))
    return bounds


def max_log_likelihood(
        xmat,
        y,
        params_init,
        fixed_p=True,
        mins_list=None,
        maxs_list=None):
    """
    Perform maximum likelihood optimization on hyperparameters thetas and ps using newton L-BFGS-B

    Args:
        xmat (numpy.ndarray) : shape = (n, k)
        y (numpy.ndarray) : shape = (n, 1)
        params_vec (numpy.ndarray) : shape = (2*k, )
        fixed_p (bool) : if set to True, p paramaters are fixed at their initial value

    Returns:
        scipy.optimize.optimize.OptimizeResult. The result of optimization as a scipy result object
    """

    if fixed_p:
        dim = int(params_init.shape[0] / 2)
        p_vec = params_init[dim:]

        def minus_llk_opti(theta_vec):
            params_fixedp = np.concatenate((np.array(theta_vec), p_vec))
            print(theta_vec)
            return - log_likelihood(xmat, y, params_fixedp)
        if mins_list:
            bounds = bounds_generator(mins_list, maxs_list)
        else:
            bounds = None
        opti = optimize.minimize(
            fun=minus_llk_opti, x0=params_init[0:dim], bounds=bounds, method="L-BFGS-B")
    else:
        def minus_llk_opti(params):
            print(params)
            return - log_likelihood(xmat, y, params)
        if mins_list:
            bounds = bounds_generator(mins_list, maxs_list)
        else:
            bounds = None
        opti = optimize.minimize(
            fun=minus_llk_opti,
            x0=params_init,
            method="L-BFGS-B",
            bounds=bounds)
    return opti

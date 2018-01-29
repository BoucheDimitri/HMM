import numpy as np

import test_functions as test_func
import max_likelihood as max_llk
import bayesian_optimization as bayes_opti
import AcqFuncs as AF
import initializations as initial


# Number of initial points
n = 5

# Number of iterations for bayesian optimization
n_it = 5

# Choice of objective function
objective_func = test_func.test_1d

# Choice of acquisition function
acq_func = AF.ExpImpr(xi=0.01)

# Choice domain for ojective
bounds = (0, 10)

# Set the constratins for optimization
constraints = ({'type': 'ineq', 'fun': lambda x: x},
               {'type': 'ineq', 'fun': lambda x: -x+10})

# Random initialization of points
xmat = initial.xmat_inbounds(n, [bounds])
y = initial.init_y(xmat, objective_func)

# Initialization of kernel parameters
theta_vec = np.array([10])
p_vec = np.array([2])

# Parameters in one vector for mle
params_init = np.concatenate((theta_vec, p_vec))

# MLE for theta and p
opti = max_llk.max_log_likelihood(
        xmat,
        y,
        params_init,
        fixed_p=True,
        mins_list=[0.01],
        maxs_list=[None])
#theta_vec = opti.x


x, y = bayes_opti.bayesian_opti_plot_1d(xmat,
                  y,
                  n_it,
                  theta_vec,
                  p_vec,
                  acq_func,
                  objective_func,
                  [bounds])

# R = exp_kernel.kernel_mat(xtest, theta_vec, p_vec)
# xnew = np.random.rand(1)
# rx = exp_kernel.kernel_rx(xtest, xnew, theta_vec, p_vec)
# print(rx)
#
# #Y
# y = np.zeros((n, 1))
# for i in range(0, n):
#     y[i, 0] = test_func.test_1d(xtest[i, :])
#
#
#
# # Test for cho_inv
# Rinv = cho_inv.cholesky_inv(R)
# #print(np.dot(Rinv, R))
# #Rinv = np.linalg.inv(R)
#
#
# # Test for prediction_formulae
# beta = pred.beta_est(y, Rinv)
# #print(beta)
# y_hat = pred.y_est(rx, y, Rinv, beta)
# # print(y_hat)
# sighat = pred.hat_sigmaz_sqr(y, Rinv, beta)
# # print(sighat)
# sigma_sq_pred = pred.sigma_sqr_est(y, rx, Rinv, beta)
# # print(sigma_sq_pred)
#
#
# fmin = np.min(y)
# hat_sigma = np.power(sigma_sq_pred, 0.5)
# exp_impr = AF.ExpImpr(xi=0.01, fmin=fmin)
#
# bounds = (0, 10)
#
# axes = viz.bayes_opti_plot_1d(xtest,
#                        y,
#                        Rinv,
#                        beta,
#                        theta_vec,
#                        p_vec,
#                        bounds,
#                        grid_size=1000,
#                        acq_func=exp_impr,
#                        objective_func=test_func.test_1d)
# plt.show()
#
# # nit = 5
# #
# # for i in range(0, nit):
# #     exp_impr = AF.ExpImpr(xi=0, fmin=fmin)
# #     print (exp_impr.evaluate(y_hat, hat_sigma))
#
#
# # Test for bayesian optimization
# # xbest = bayes_opti.bayesian_search(xtest, y, theta_vec, p_vec, xnew, exp_impr)
# # print(xbest)
# #
# # n_it = 10
# # xx, yy = bayes_opti.bayesian_opti(xtest, y, n_it,
# #                                   theta_vec,
# #                                   p_vec,
# #                                   exp_impr,
# #                                   test_func.test_1d,
# #                                   bounds=[(0, 10)])
# #
# # print(xx)
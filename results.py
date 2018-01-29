# *Third party imports*
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# *Local imports*
import cho_inv
import exp_kernel
import test_functions as test_func
import prediction_formulae as pred
import AcqFuncs as AF
import visualization as viz
import initializations as initial
import max_likelihood as max_llk
import bayesian_optimization as bayes_opti
import metrics
from decimal import *
import copy

# **EXECUTION PARAMETERS**

#number of iteration of the whole optimization
nb_opti = 10

# number of points selected randomly for initial sampling
n = 10

# number of dimensions
d = 2

# number of bayesian optimization iterations
n_it = 90

# Choice of objective function
objective_func = test_func.mystery_vec

# Choice of acquisition function
# acq_func1 will be used, acq_func2 is only for plotting to draw comparisons
acq_func1 = AF.ExpImpr(xi=0.01)
acq_func2 = AF.LowConfBound(eta=2)

# Choice domain for ojective
bounds = ((0, 5), (0, 5))

# Should max likelihood be performed  for kernel params ?
perform_mle = True

# Should acquisition function be plotted at initial step ?
plot_acq_2d = True

#initialize results
table_EI = []

# Parameters for performance metrics
gridsize = [30,30]
true_y_sol=-1.4565
true_x_sol=[2.5044,2.5778]
p=15/100

for it in range(0,nb_opti):

    # **EXECUTION**

    # Random initialization of points
    xmat = initial.xmat_inbounds(n, bounds)
    y = initial.init_y(xmat, objective_func)
    
    # Initialization of kernel parameters
    theta_vec = np.array([10, 10])
    p_vec = np.array([1.9, 1.9])
    
    # Parameters in one vector for mle
    params_init = np.concatenate((theta_vec, p_vec))
    
    # MLE for theta and p
#    if perform_mle:
#        opti = max_llk.max_log_likelihood(
#            xmat,
#            y,
#            params_init,
#            fixed_p=False,
#            mins_list=[0.01, 0.01, 0.1, 0.1],
#            maxs_list=[None, None, 1.99, 1.99])
#        print(opti)
#        theta_vec = opti.x[0:d]
#        p_vec = opti.x[d:]
    
    # Plot of initial acquisition function in 2d
    if plot_acq_2d and (d == 2):
        # Computation of the necessaries quantities
        R = exp_kernel.kernel_mat(xmat, theta_vec, p_vec)
        Rinv = cho_inv.cholesky_inv(R)
        beta = pred.beta_est(y, Rinv)
        # Plot acq_func1
        viz.plot_acq_func_2d(xmat,
                             y,
                             Rinv,
                             beta,
                             theta_vec,
                             p_vec,
                             bounds,
                             (100, 100),
                             acq_func1)
        # Plot acq_func2
        viz.plot_acq_func_2d(xmat,
                             y,
                             Rinv,
                             beta,
                             theta_vec,
                             p_vec,
                             bounds,
                             (100, 100),
                             acq_func2)
        # Plot objective_func
        ax = viz.add_points_2d(xmat, y)
        ax = viz.plot_func_2d(
            bounds,
            (100,
             100),
            test_func.mystery_vec,
            plot_type="ColoredSurface",
            alpha=0.5,
            ax=ax,
            title="Objective function")
        plt.show()
    
    #Bayesian optimization
    xmat_opti, y_opti = bayes_opti.bayesian_opti(xmat, y, n_it,
                                                 theta_vec,
                                                 p_vec,
                                                 acq_func1,
                                                 objective_func,
                                                 bounds=bounds)
    
    #Metrics of performance
    table_EI += [metrics.all_metrics(objective_func, bounds, gridsize,
                    y_opti[n:], xmat_opti[n:,:], true_y_sol, true_x_sol, theta_vec, p_vec, p)]

table_EI0 = copy.deepcopy(table_EI)
nb_simu_rejected_EI = 0
all_results = []
for j in table_EI:
    if j[1]==0 or j[2]==0:
        nb_simu_rejected_EI += 1
table_EI = [j for j in table_EI if j[1]>0 and j[2]>0 ]
results_brut_EI = np.array(table_EI)
results_mean_EI = np.mean(results_brut_EI, axis=0)
results_mean_EI = list(results_mean_EI)
results_mean_EI[0]="%.4f" %results_mean_EI[0]
results_mean_EI[1]=str(np.int(results_mean_EI[1]))
results_mean_EI[2]=str(np.int(results_mean_EI[2]))
results_mean_EI[3]="%.4f" %results_mean_EI[3]
results_mean_EI[4]="%.4f" %results_mean_EI[4]
results_mean_EI.insert(0,"EI")    

#######

#initialize results
table_LBC = []

for it in range(0,nb_opti):

    # **EXECUTION**
    
    
    # Random initialization of points
    xmat = initial.xmat_inbounds(n, bounds)
    y = initial.init_y(xmat, objective_func)
    
    # Initialization of kernel parameters
    theta_vec = np.array([10, 10])
    p_vec = np.array([1.9, 1.9])
    
    # Parameters in one vector for mle
    params_init = np.concatenate((theta_vec, p_vec))
    
    # MLE for theta and p
#    if perform_mle:
#        opti = max_llk.max_log_likelihood(
#            xmat,
#            y,
#            params_init,
#            fixed_p=False,
#            mins_list=[0.01, 0.01, 0.1, 0.1],
#            maxs_list=[None, None, 1.99, 1.99])
#        print(opti)
#        theta_vec = opti.x[0:d]
#        p_vec = opti.x[d:]
    
    # Plot of initial acquisition function in 2d
    if plot_acq_2d and (d == 2):
        # Computation of the necessaries quantities
        R = exp_kernel.kernel_mat(xmat, theta_vec, p_vec)
        Rinv = cho_inv.cholesky_inv(R)
        beta = pred.beta_est(y, Rinv)
        # Plot acq_func1
        viz.plot_acq_func_2d(xmat,
                             y,
                             Rinv,
                             beta,
                             theta_vec,
                             p_vec,
                             bounds,
                             (100, 100),
                             acq_func1)
        # Plot acq_func2
        viz.plot_acq_func_2d(xmat,
                             y,
                             Rinv,
                             beta,
                             theta_vec,
                             p_vec,
                             bounds,
                             (100, 100),
                             acq_func2)
        # Plot objective_func
        ax = viz.add_points_2d(xmat, y)
        ax = viz.plot_func_2d(
            bounds,
            (100,
             100),
            test_func.mystery_vec,
            plot_type="ColoredSurface",
            alpha=0.5,
            ax=ax,
            title="Objective function")
        plt.show()
    
    #Bayesian optimization
    xmat_opti, y_opti = bayes_opti.bayesian_opti(xmat, y, n_it,
                                                 theta_vec,
                                                 p_vec,
                                                 acq_func2,
                                                 objective_func,
                                                 bounds=bounds)
    
    #Metrics of performance
    table_LBC += [metrics.all_metrics(objective_func, bounds, gridsize,
                    y_opti[n:], xmat_opti[n:,:], true_y_sol, true_x_sol, theta_vec, p_vec, p)]

table_LBC0 = copy.deepcopy(table_LBC)
nb_simu_rejected_LBC = 0
all_results = []
for j in table_LBC:
    if j[1]==0 or j[2]==0:
        nb_simu_rejected_LBC += 1
table_LBC = [j for j in table_LBC if j[1]>0 and j[2]>0 ]
results_brut_LBC = np.array(table_LBC)
results_mean_LBC = np.mean(results_brut_LBC, axis=0)
results_mean_LBC = list(results_mean_LBC)
results_mean_LBC[0]="%.4f" %results_mean_LBC[0]
results_mean_LBC[1]=str(np.int(results_mean_LBC[1]))
results_mean_LBC[2]=str(np.int(results_mean_LBC[2]))
results_mean_LBC[3]="%.4f" %results_mean_LBC[3]
results_mean_LBC[4]="%.4f" %results_mean_LBC[4]
results_mean_LBC.insert(0,"LBC")

#######

df = pd.DataFrame([results_mean_EI,results_mean_LBC], columns = ["Criterion","Best min", "f%", "x%", "x*", "RMS"])
print(df.to_latex(index=None))
nb_simu_rejected_EI
nb_simu_rejected_LBC
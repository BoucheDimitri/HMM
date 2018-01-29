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

#all results for all param
all_table = []

#Set of parameters for eta
eta_vec = [1, 1.5, 2, 2.5, 3]

#number of iteration of the whole optimization
nb_opti = 10

# number of points selected randomly for initial sampling
n = 10

# number of dimensions
d = 2

# number of bayesian optimization iterations
n_it = 40

# Choice of objective function
objective_func = test_func.mystery_vec

# Choice domain for ojective
bounds = ((0, 5), (0, 5))

# Should max likelihood be performed  for kernel params ?
perform_mle = True

# Should acquisition function be plotted at initial step ?
plot_acq_2d = True

# Parameters for performance metrics
gridsize = [30,30]
true_y_sol=-1.4565
true_x_sol=[2.5044,2.5778]
p=15/100    

for index_eta in range(0, len(eta_vec)):
    # **EXECUTION PARAMETERS**
    
    eta = eta_vec[index_eta]
    
    
    # Choice of acquisition function
    # acq_func1 will be used, acq_func2 is only for plotting to draw comparisons
    acq_func2 = AF.LowConfBound(eta=eta)
    

    
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
        if perform_mle:
            opti = max_llk.max_log_likelihood(
                xmat,
                y,
                params_init,
                fixed_p=False,
                mins_list=[0.01, 0.01, 0.1, 0.1],
                maxs_list=[None, None, 1.99, 1.99])
            print(opti)
            theta_vec = opti.x[0:d]
            p_vec = opti.x[d:]
        
        # Plot of initial acquisition function in 2d
        if plot_acq_2d and (d == 2):
            # Computation of the necessaries quantities
            R = exp_kernel.kernel_mat(xmat, theta_vec, p_vec)
            Rinv = cho_inv.cholesky_inv(R)
            beta = pred.beta_est(y, Rinv)

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
    
    all_table += [table_LBC]

all_table0 = copy.deepcopy(all_table)
#all_table = copy.deepcopy(all_table0)
nb_simu_rejected = np.zeros(len(eta_vec))
all_results = []
for i in range(0,len(eta_vec)):
    for j in all_table[i]:
        if j[1]==0 or j[2]==0:
            nb_simu_rejected[i] += 1
    all_table[i] = [j for j in all_table[i] if j[1]>0 and j[2]>0 ]
    results_brut_LBC = np.array(all_table[i])
    results_mean_LBC = np.mean(results_brut_LBC, axis=0)
    results_mean_LBC = list(results_mean_LBC)
    results_mean_LBC[0]="%.4f" %results_mean_LBC[0]
    results_mean_LBC[1]=str(np.int(results_mean_LBC[1]))
    results_mean_LBC[2]=str(np.int(results_mean_LBC[2]))
    results_mean_LBC[3]="%.4f" %results_mean_LBC[3]
    results_mean_LBC[4]="%.4f" %results_mean_LBC[4]
    results_mean_LBC.insert(0,eta_vec[i])
    all_results += [results_mean_LBC]

#######
df = pd.DataFrame(all_results, columns = ["$\eta$","Best min", "f%", "x%", "x*", "RMS"])
print(df.to_latex(index=None))


list(nb_simu_rejected)
name_col = []
for i in range(0,len(eta_vec)):
    name_col += ["$\eta="+str(eta_vec[i])+"$"]
df1 = pd.DataFrame([list(nb_simu_rejected)], columns = name_col)
print(df1.to_latex(index=None))

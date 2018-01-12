# nous faisons les importations necessaires
import matplotlib.pyplot as plt
import matplotlib
import numpy as np


import BootstrapFilter as bootstrap
import ResampleMove as resamplemove
import ResampleTauMove as resampletaumove
import DataGenerator as datagenerator
import TauMove as taumove
from MSE import *
from save_data import *

eta = 0.005 # std of noise in measurement
tau = 100000 #tau is such that math.sqrt(1/tau) is the std for speeds
N = 100 #N is the number of particles
T = 100 #T is the number of periods

#Initial conditions
x0 = 3
y0 = 5
xp0 = 0.03
yp0 = -0.03

mux = 0
muy = 0
mprior = [x0 + mux, y0 + muy, xp0, yp0]
stdprior = [0.0000001, 0.0000001, 0.0000001, 0.0000001]


d0 = 2
c0 = (d0-1)*100/tau #this initilization gives a first biaised estimation of tau (/100)
c0 = (2-1)/tau
locpriormean = [x0 + mux, y0 + muy]
locpriorstd = [0.0000001, 0.0000001]
speedpriormean = [xp0, yp0]
speedpriorstd = [0.0000001, 0.0000001]

#calcul de la variance de l'estimateur

def resample_move_with_estimation_of_tau_n_times(n):

    all_est_tau_post = []
    all_var_tau_post = []
    
    for iteration_algo in range(0,n):
        
        data = datagenerator.loc_data(x0, y0, xp0, yp0, T, tau, eta)
        zs = data["z"].as_matrix()
        
        allparticlesrtm, allweightsrtm, alltaurtm = resampletaumove.resample_tau_move(locpriormean,
                          locpriorstd,
                          speedpriormean,
                          speedpriorstd,
                          zs,
                          N,
                          eta,
                          d0,
                          c0,
                          restype = "stratified")
        
        ##### PLOT #####
        
        plt.figure()
        plt.plot(data["x"], data["y"], label="True trajectory")
        means = resamplemove.extract_loc_means(allparticlesrtm)
        plt.scatter(means[0], means[1], label="Particle means")
        plt.legend()
        
        ################

        alltau_plot = np.array(alltaurtm[1:])
        allweights = np.array(allweightsrtm)
        alltau_weighted = alltau_plot*allweights
        
        est_tau_post = alltau_weighted.mean(axis=1)
        all_est_tau_post.append(est_tau_post)
        var_tau_post = alltau_weighted.var(axis=1)
        all_var_tau_post.append(var_tau_post)
        
        print(iteration_algo+1, "ème itération")
        
    return(all_est_tau_post, all_var_tau_post)
    
all_est_tau_post, all_var_tau_post = resample_move_with_estimation_of_tau_n_times(100)

all_est_tau_post = np.array(all_est_tau_post)
var_all_est_tau_post = all_est_tau_post.var(axis=0)

all_var_tau_post = np.array(all_var_tau_post)
mean_var_post = all_var_tau_post.mean(axis=0)

plt.figure()
plt.plot(var_all_est_tau_post)
plt.title("Variance de la loi a posteriori à chaque étape")
plt.legend()

relative_error = var_all_est_tau_post/mean_var_post 

plt.figure()
plt.plot(relative_error)
plt.title("Variance de l'estimateur paticulaire à chaque étape")
plt.legend()
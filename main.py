import matplotlib.pyplot as plt
import matplotlib
import numpy as np


import BootstrapFilter as bootstrap
import ResampleMove as resamplemove
import ResampleTauMove as resampletaumove
import DataGenerator as datagenerator


#Change matplotlib fontsize on graphs
matplotlib.rcParams.update({'font.size': 15})


######PARAMETERS######################################################
#eta = std of noise in measurement
eta = 0.005
#tau is such that math.sqrt(1/tau) is the std for speeds
tau = 1000000
#N is the number of particles
N = 1000
#T is the number of periods
T = 50
#Initial conditions
x0 = 3
y0 = 5
xp0 = 0.002
yp0 = -0.013
#noise on initial position
mux = 0.001
muy = -0.1
mprior = [x0 + mux, y0 + muy, 0.002, -0.013]
stdprior = [0.04, 0.4, 0.003, 0.003]



#############GENERATE####DATA###########################################
data = datagenerator.loc_data(x0, y0, xp0, yp0, T, tau, eta)



####BOOTSTRAP###FILTER##################################################
zs = data["z"].as_matrix()[1:]
allparticlesbs, allweightsbs = bootstrap.bootstrap_filter(mprior,
                                                      stdprior,
                                                      zs,
                                                      N,
                                                      tau,
                                                      eta,
                                                      "stratified")


#Compute mean of particles
means = np.array([np.mean(a, axis=0) for a in allparticlesbs])

#Plot the result
plt.figure()
plt.plot(data["x"][1:], data["y"][1:], label="real_trajectory")
plt.scatter(means[:, 0], means[:, 1], label="particle_means")
plt.legend()
plt.xlabel("x coordinate")
plt.ylabel("y coordinate")
plt.title("Boostrap filter; 1/tau="
          + str(1/tau)
          + "; eta="
          + str(eta)
          +"; N="
          + str(N)
          +" ;T="
          + str(T))





######################RESAMPLE###MOVE###WITH###FIXED###TAU####################
locpriormean = [x0 + mux, y0 + muy]
locpriorstd = [0.04, 0.4]
speedpriormean = [xp0, yp0]
speedpriorstd = [0.001, 0.001]

zs = data["z"].as_matrix()

allparticlesrm, allweightsrm = resamplemove.resample_move(locpriormean,
                                         locpriorstd,
                                         speedpriormean,
                                         speedpriorstd,
                                         zs,
                                         N,
                                         tau,
                                         eta,
                                         0.995,
                                         1e-6,
                                         1e-6,
                                        10,
                                        movetype="noninformative",
                                         restype="stratified")

plt.figure()
plt.plot(data["x"], data["y"], label="True trajectory")
means = resamplemove.extract_loc_means(allparticlesrm)
plt.scatter(means[0], means[1], label="Particle means")
varw = [np.var(w) for w in allweightsrm]

plt.legend()



######################RESAMPLE###MOVE###WITH###ESTIMATION###OF###TAU##########
d0 = 1
c0 = tau #this initialization gives a first unbiased estimation of tau
locpriormean = [x0 + mux, y0 + muy]
locpriorstd = [0.04, 0.4]
speedpriormean = [xp0, yp0]
speedpriorstd = [0.001, 0.001]

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

plt.figure()
plt.plot(data["x"], data["y"], label="True trajectory")
means = resamplemove.extract_loc_means(allparticlesrtm)
plt.scatter(means[0], means[1], label="Particle means")
varw = [np.var(w) for w in allweightsrtm]

plt.legend()

alltau_plot = np.array(alltaurtm[1:])
allweights = np.array(allweightsrtm)
alltau_weighted = alltau_plot*allweights

var_tau_post = alltau_weighted.var(axis=1)

plt.figure()
plt.plot(var_tau_post, label="Variance of simulations of tau")
plt.legend()

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
plt.plot(var_all_est_tau_post, label="Variance of estimations of tau for each step")
plt.legend()

relative_error = var_all_est_tau_post/mean_var_post 

plt.figure()
plt.plot(relative_error, label="Relative error of estimations of tau for each step")
plt.legend()
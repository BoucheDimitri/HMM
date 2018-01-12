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

#Change matplotlib fontsize on graphs
matplotlib.rcParams.update({'font.size': 10})

Q = 48

for i in range(0, Q):

    #################################
    ### PARAMETER INITIALISATIONS ###
    #################################
    
    eta = 0.005 # std of noise in measurement
    tau = 1/0.000001 #tau is such that math.sqrt(1/tau) is the std for speeds
    N = 100 #N is the number of particles
    T = 150 #T is the number of periods
    
    #Initial conditions
    x0 = 0.01
    y0 = 20
    xp0 = 0.002
    yp0 = -0.06
    #noise on initial position
    mux = 0
    muy = 0
    mprior = [x0 + mux, y0 + muy, xp0, yp0]
    stdprior = [0.0000001, 0.0000001, 0.0000001, 0.0000001]
    
    
    #######################
    ### DATA GENERATION ###
    #######################
    
    # nous générons l'ensemble des données grâce à la fonction loc_data
    # data est un dataframe de cinq colonnes: x, y, xp, yp, et z
    data = datagenerator.loc_data(x0, y0, xp0, yp0, T, tau, eta)
    
    
    ########################
    ### BOOTSTRAP FILTER ###
    ########################
    
    # A présent nous allons observer les performances du filtre Bootstrap appliqué à notre problème
    # nous stockons les données observés dans zs
    zs = data["z"].as_matrix()[1:]
    # nous appliquons notre filtre bootstrap et obtenons l'ensemble de nos particules et de nos poids
    # en paramètre nous y entrons nos prior des moyennes et std, la variable observé zs, le nombre de particules, tau et eta
    allparticlesbs, allweightsbs = bootstrap.bootstrap_filter(mprior, stdprior, zs, N, tau, eta, "stratified")
    
    # nous calculons la moyenne de nos particules et nous les stockons
    means = np.array([np.mean(a, axis=0) for a in allparticlesbs])
    X_bs = means[:, 0]
    Y_bs = means[:, 1]
    
    
    ####################################
    ### RESAMPLE MOVE WITH FIXED TAU ###
    ####################################
    
    locpriormean = [x0 + mux, y0 + muy]
    locpriorstd = [0.0000001, 0.0000001]
    speedpriormean = [xp0, yp0]
    speedpriorstd = [0.0000001, 0.0000001]
    
    zs = data["z"].as_matrix()
    
    allparticlesrm, allweightsrm = resamplemove.resample_move(locpriormean, locpriorstd, speedpriormean, speedpriorstd, zs, N, tau, eta, 0.995, 1e-6, 1e-6, 10, movetype="noninformative", restype="stratified")
    
    means = resamplemove.extract_loc_means(allparticlesrm)
    X_rmft = np.array(means[0])
    Y_rmft = np.array(means[1])
    
    
    
    ############################################
    ### RESAMPLE MOVE WITH ESTIMATION OF TAU ###
    ############################################
    
    # a présent nous allons appliquer le resample move algorithm, mais au lieu d'avoir tau en paramètre fixe comme précedemment, nous allons devoir l'estimer
    
    # nous initialisons nos paramètres prior
    d0 = 2
    c0 = (d0-1)*100/tau #this initilization gives a first biaised estimation of tau (/100)
    c0 = (2-1)/tau
    locpriormean = [x0 + mux, y0 + muy]
    locpriorstd = [0.0000001, 0.0000001]
    speedpriormean = [xp0, yp0]
    speedpriorstd = [0.0000001, 0.0000001]
    
    # nous stockons nos observations
    zs = data["z"].as_matrix()
    
    # nous appliquons notre filtre 
    allparticlesrtm, allweightsrtm, alltaurtm = resampletaumove.resample_tau_move(locpriormean, locpriorstd, speedpriormean, speedpriorstd, zs, N, eta, d0, c0, restype = "stratified")
    
    # nous stockons nos résultats
    means = resamplemove.extract_loc_means(allparticlesrtm)
    X_rm = np.array(means[0])
    Y_rm = np.array(means[1])
    
    # nous affichons les estimations de tau
    alltaurtm = np.array(alltaurtm)
    all_tau_estimations = alltaurtm.mean(axis=1)
    
    # et les variances e tau
    alltau_plot = np.array(alltaurtm[1:])
    allweights = np.array(allweightsrtm)
    sum_weights = allweights.sum(axis=1)
    alltau_weighted = alltau_plot*allweights
    var_tau_post = alltau_weighted.var(axis=1)
    
    
    ###################
    ### SAVING DATA ###
    ###################
    
    save_data(data, [X_bs, Y_bs], [X_rmft, Y_rmft], [X_rm, Y_rm], all_tau_estimations, var_tau_post)
    
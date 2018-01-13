################
### PACKAGES ###
################

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd

import BootstrapFilter as bootstrap
import ResampleMove as resamplemove
import ResampleTauMove as resampletaumove
import DataGenerator as datagenerator
import TauMove as taumove
from MSE import *
from save_data import *

def all_algorithms_Qtimes(Q, eta, tau, N, T, x0, y0, xp0, yp0, mux, muy, mprior, stdprior):
    
    #initialisation
    locpriormean = [x0 + mux, y0 + muy]
    locpriorstd = [0.0000001, 0.0000001]
    speedpriormean = [xp0, yp0]
    speedpriorstd = [0.0000001, 0.0000001]
    
    #Change matplotlib fontsize on graphs
    matplotlib.rcParams.update({'font.size': 10})
    
    #listes pour stocker les résultats
    all_outputs = []
    
    for i in range(0, Q):
        
        columns = ["X", "Y", "Z", "X_bs", "Y_bs", "X_rmft", "Y_rmft", "X_rm", "Y_rm", "tau_rm", "var_tau_rm"]
        output = pd.DataFrame(columns = columns)
        
        #######################
        ### DATA GENERATION ###
        #######################
    
        # nous générons l'ensemble des données grâce à la fonction loc_data
        # data est un dataframe de cinq colonnes: x, y, xp, yp, et z
        data = datagenerator.loc_data(x0, y0, xp0, yp0, T, tau, eta)
        output["X"] = data["x"]
        output["Y"] = data["y"]
        output["Z"] = data["z"]
        
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
        output["X_bs"] = X_bs
        output["Y_bs"] = Y_bs
        
        
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
        output["X_rmft"] = X_rmft
        output["Y_rmft"] = Y_rmft
        
        
        ############################################
        ### RESAMPLE MOVE WITH ESTIMATION OF TAU ###
        ############################################
        
        # a présent nous allons appliquer le resample move algorithm, mais au lieu d'avoir tau en paramètre fixe comme précedemment, nous allons devoir l'estimer
        
        # nous initialisons nos paramètres prior
        d0 = 2
        c0 = (d0-1)/tau
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
        output["X_rm"] = X_rm
        output["Y_rm"] = Y_rm
        # nous affichons les estimations de tau
        alltaurtm = np.array(alltaurtm)
        all_tau_estimations = np.append(0,alltaurtm.mean(axis=1))
        output["tau_rm"] = all_tau_estimations
        
        # et les variances de tau
        alltau_plot = np.array(alltaurtm[1:])
        allweights = np.array(allweightsrtm)
        sum_weights = allweights.sum(axis=1)
        alltau_weighted = alltau_plot*allweights
        var_tau_post = np.append(np.zeros(2), alltau_weighted.var(axis=1))
        output["var_tau_rm"] = var_tau_post
        
        # concaténation des résultats
        all_outputs.append(output)
        
    #sortie
    return(all_outputs)

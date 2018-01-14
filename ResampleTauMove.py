import numpy as np
import math
from scipy.stats import invgamma

import Probas as probas
import Resampling as resampling
import AlgebraTools as algtools
import Moves as moves
import TauMove as taumove
import ResampleMove as resamplemove

def augment_all_particles_with_estimated_tau(particles, tau_moved):
    """
    Perform the augment_one_particle function
    on all rows of an array of particles
    :param particles: numpy array
    :param tau_moved: vector of estimates of tau for each particle
    :return: numpy array, augmented array of particles
    """
    npartis = particles.shape[0]
    ndims = particles.shape[1]
    augmented = np.zeros((npartis, ndims + 2))
    for i in range(0, npartis):
        augmented[i, :] = resamplemove.augment_one_particle(particles[i, :], tau_moved[i])
    return augmented

def all_logweights_with_estimated_tau(particles, zk, tau_moved, eta):
    """
    Compute weights as done on one particle in
    one_logweight for all particles
    :param particles: numpy array, array of particles
    :param zk: float, bearing observation
    :param previoustau: estimation of inverse of variance of speeds
    :param eta: std of noise on bearings measurement
    :return: numpy array,
    the array of log of weight for the particle up to an additive constant
    """
    npartis = particles.shape[0]
    lw = np.zeros((npartis, ))
    for i in range(0, npartis):
        lw[i] = probas.one_logweight(particles[i, :], zk, tau_moved[i], eta)
    return lw

def resample_tau_move_iteration(previouspartis,
                            tau_moved,
                            zs,
                            eta,
                            d0,
                            c0,
                            restype):
    """
    An iteration of resample move algorithm with only a tau-move
    :param previouspartis: numpy array : the array of particles
    :param previouspartis: numpy array : the array of estimations of tau
    :param zs: numpy array : the vector of bearings
    :param eta: float, std of noise on bearings observations
    :param d0, c0: float: initialization parameters for distribution of tau
    :param restype: str, type of resampleing, either "stratified" of "multinomial"
    :return: tuple, (array of resampled and moved particles, normalized weights used)
    """
    augmented = augment_all_particles_with_estimated_tau(previouspartis, tau_moved)
    t = augmented.shape[1]//2 - 1
    lw = all_logweights_with_estimated_tau(augmented, zs[t], tau_moved, eta)
    normw = probas.norm_exp_logweights(lw)
    if restype == "stratified":
        resampled = resampling.stratified_resampling(augmented, normw)
    else:
        resampled = resampling.multi_resampling(augmented, normw)
    tau_moved = taumove.pertub_all_tau(previouspartis, d0, c0)
    return resampled, normw, tau_moved

def resample_tau_move(locmean,
                  locstd,
                  speedmean,
                  speedstd,
                  zs,
                  N,
                  eta,
                  d0,
                  c0,
                  restype):
    """
    Performs resample move
    :param locmean: list, [mean loc x, mean loc y] for initialization
    :param locstd: list, [std loc x, std loc y] for initialization
    :param speedmean: list, [mean speed x, mean speed y] for initialization
    :param speedstd: list, [std speed x, std speed y] for initialization
    :param zs: numpy array : the vector of bearings
    :param N : int, the number of particles
    :param eta: float, std of noise on bearings observations
    :param d0, c0: float, initialization parameters for estimation of tau
    :param restype: str, type of resampleing, either "stratified" of "multinomial"
    :return: tuple, (list of particles generated, list of normalized weights)
    """
    allparticles = []
    allweights = []
    alltau = []
    T = zs.shape[0]
    particles0 = resamplemove.init_locs(locmean, locstd, N)
    allparticles.append(particles0)
    particles1 = resamplemove.init_speeds(particles0,
                             speedmean,
                             speedstd,
                             N)
    allparticles.append(particles1)
    tau_moved = 1/invgamma.rvs(d0, scale=c0, size=N)
    alltau.append(tau_moved)
    for t in range(1, T-1):
        newparticles, weights, newtau = resample_tau_move_iteration(
            particles1,
            tau_moved,
            zs,
            eta,
            d0,
            c0,
            restype)
        allparticles.append(newparticles)
        allweights.append(weights)
        alltau.append(newtau)
        del particles1
        particles1 = newparticles
        tau_moved = newtau        
        print(t)
    return allparticles, allweights, alltau

#test resample_tau_move
    '''
locpriormean = [x0 + mux, y0 + muy]
locpriorstd = [0.04, 0.4]
speedpriormean = [xp0, yp0]
speedpriorstd = [0.001, 0.001]
restype = "stratified"

zs = data["z"].as_matrix()

particles0 = resamplemove.init_locs(locpriormean, locpriorstd, N)
particles = resamplemove.init_speeds(particles0,
                             speedpriormean,
                             speedpriorstd,
                             N)

tau_moved = np.random.gamma(shape=d0, scale=c0, size=N)
tau_moved.shape[0]

particles, weights, tau_moved = resample_tau_move_iteration(
            particles, tau_moved,
            zs,
            eta,
            d0,
            c0,
            restype)

locmean = [x0 + mux, y0 + muy]
locstd = [0.04, 0.4]
speedmean = [xp0, yp0]
speedstd = [0.001, 0.001]

allparticles = []
allweights = []
alltau = []
T = zs.shape[0]
particles0 = resamplemove.init_locs(locmean, locstd, N)
allparticles.append(particles0)
particles1 = resamplemove.init_speeds(particles0,
                             speedmean,
                             speedstd,
                             N)
allparticles.append(particles1)
tau_moved = np.random.gamma(shape=d0, scale=c0, size=N)
alltau.append(tau_moved)
'''
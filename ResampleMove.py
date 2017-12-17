import numpy as np
import matplotlib.pyplot as plt
import math

import Probas as probas
import Resampling as resampling
import AlgebraTools as algtools
import Moves as moves


def augment_one_particle(particle, tau):
    """
    Augmentation step for one particle
    :param particle: numpy array, particle to augment
    :param tau: inverse of variance of speed
    :return: nuumpy array, augmented particle
    """
    x, y = algtools.particle_to_xyvecs(particle)
    xpaug = np.random.normal(x[-1], math.sqrt(1 / tau))
    ypaug = np.random.normal(y[-1], math.sqrt(1 / tau))
    x = np.append(x, xpaug)
    y = np.append(y, ypaug)
    return np.concatenate((x, y))


def augment_all_particles(particles, tau):
    """
    Perform the augment_one_particle function
    on all rows of an array of particles
    :param particles: numpy array
    :param tau: inverse of variance of speed
    :return: nuumpy array, augmented array of particles
    """
    npartis = particles.shape[0]
    ndims = particles.shape[1]
    augmented = np.zeros((npartis, ndims + 2))
    for i in range(0, npartis):
        augmented[i, :] = augment_one_particle(particles[i, :], tau)
    return augmented


def init_locs(locmean, locstd, N):
    """
    initialization of positions at t=0
    :param locmean: list, [mean loc x, mean loc y]
    :param locstd: list, [std loc x, std loc y]
    :param N: number of particles
    :return: Set of N particles of initial position
    """
    x0s = np.random.normal(locmean[0], locstd[0], (N, 1))
    y0s = np.random.normal(locmean[1], locstd[1], (N, 1))
    return np.concatenate((x0s, y0s), axis=1)


def init_speeds(initloc, speedmean, speedstd, N):
    """
    initialization of speeds at t=0, augment the initloc particles
    with speeds at t=0
    :param speedmean: list, [mean speed x, mean speed y]
    :param speedstd: list, [std speed x, std speed y]
    :param N: number of particles
    :return: Set of N particles of initial position and speed
    """
    xp0s = np.random.normal(speedmean[0], speedstd[0], (N, 1))
    yp0s = np.random.normal(speedmean[1], speedstd[1], (N, 1))
    particles = np.concatenate((initloc[:, 0].reshape(N, 1),
                                xp0s,
                                initloc[:, 1].reshape(N, 1),
                                yp0s),
                               axis=1)
    return particles


def resample_move_iteration(previouspartis,
                            zs,
                            tau,
                            eta,
                            r1,
                            r2,
                            r3,
                            a,
                            b,
                            movetype="noninformative",
                            restype="stratified"):
    """
    An iteration of resample move algorithm
    :param previouspartis: numpy array : the array of particles
    :param zs: numpy array : the vector of bearings
    :param tau: float, inverse of variance of speed
    :param eta: float, std of noise on bearings observations
    :param r1: float, r1 parameter for the rescale move
    :param r2: float, r2 parameter for the local perturbation move
    :param r3: float, r3 parameter for the local perturbation move,
    only used if movetype is set to "noninformative"
    :param a: int, index of beginning of the bloc to perturb in local perturbation
    move
    :param b: int, index of ending of the bloc to perturb in local perturbation
    move
    :param movetype: str, if set to noninformative, local perturbations
    are done independantly for x and y and without taking z into account
    in the distribution
    :param restype: str, type of resampleing, either "stratified" of "multinomial"
    :return: tuple, (array of resampled and moved particles, normalized weights used)
    """
    augmented = augment_all_particles(previouspartis, tau)
    t = augmented.shape[1]//2 - 1
    lw = probas.all_logweights(augmented, zs[t], tau, eta)
    normw = probas.norm_exp_logweights(lw)
    if restype == "stratified":
        resampled = resampling.stratified_resampling(augmented, normw)
    else:
        resampled = resampling.multi_resampling(augmented, normw)
    rescaled = moves.rescale_all_particles(r1, resampled, zs, tau, eta)
    if movetype == "noninformative":
        perturbed = moves.perturb_all_particles_noninf(rescaled, zs, a, b, r2, r3, tau, eta)
    else:
        perturbed = moves.pertub_all_particles(rescaled, zs, a, b, r2, tau, eta)
    return perturbed, normw


def resample_move(locmean,
                  locstd,
                  speedmean,
                  speedstd,
                  zs,
                  N,
                  tau,
                  eta,
                  r1,
                  r2,
                  r3,
                  blocksize,
                  movetype,
                  restype):
    """
    Performs resample move
    :param locmean: list, [mean loc x, mean loc y] for initialization
    :param locstd: list, [std loc x, std loc y] for initialization
    :param speedmean: list, [mean speed x, mean speed y] for initialization
    :param speedstd: list, [std speed x, std speed y] for initialization
    :param zs: numpy array : the vector of bearings
    :param N : int, the number of particles
    :param tau: float, inverse of variance of speed
    :param eta: float, std of noise on bearings observations
    :param r1: float, r1 parameter for the rescale move
    :param r2: float, r2 parameter for the local perturbation move
    :param r3: float, r3 parameter for the local perturbation move,
    only used if movetype is set to "noninformative"
    :param blocksize: int, size of the blocks for local perturbation,
    as advised in article, only the last blocksize speeds are perturbed
    at each iteration
    :param movetype: str, if set to noninformative, local perturbations
    are done independantly for x and y and without taking z into account
    in the distribution
    :param restype: str, type of resampleing, either "stratified" of "multinomial"
    :return: tuple, (list of particles generated, list of normalized weights)
    """
    allparticles = []
    allweights = []
    T = zs.shape[0]
    particles0 = init_locs(locmean, locstd, N)
    allparticles.append(particles0)
    particles1 = init_speeds(particles0,
                             speedmean,
                             speedstd,
                             N)
    allparticles.append(particles1)
    for t in range(1, T-1):
        newparticles, weights = resample_move_iteration(
            particles1,
            zs,
            tau,
            eta,
            r1,
            r2,
            r3,
            a=max(t-blocksize, 0),
            b=t,
            movetype=movetype,
            restype=restype)
        allparticles.append(newparticles)
        allweights.append(weights)
        del particles1
        particles1 = newparticles
        print(t)
    return allparticles, allweights


def extract_loc_means(allparticles):
    """
    Compute list of mean localization from particles
    :param allparticles: list of particles
    :return: list of mean of localization
    """
    locmeansx = []
    locmeansy = []
    for particle in allparticles:
        meanparticle = np.mean(particle, axis=0)
        x, y = algtools.particle_to_xyvecs(meanparticle)
        xloc = x[0] + np.sum(x[1:])
        yloc = y[0] + np.sum(y[1:])
        locmeansx.append(xloc)
        locmeansy.append(yloc)
    return locmeansx, locmeansy











import numpy as np
import math
import Resampling as resampling
import DataGenerator as datagenerator
import Probas as probas
import importlib
import matplotlib.pyplot as plt
import math



def logprop_gauss_ppf(x, m, sigma):
    return -(x-m)*(x-m)/(2*sigma*sigma)


def one_logweight(particle, z, eta):
    m = math.atan(particle[1]/particle[0])
    lw = logprop_gauss_ppf(z, m, eta)
    return lw


def all_logweights(particles, z, eta):
    npartis = particles.shape[0]
    lw = np.zeros((npartis, ))
    for i in range(0, npartis):
        lw[i] = one_logweight(particles[i, :], z, eta)
    return lw


def transition(particle, tau):
    """
    Move a particle according to transition kernel
    :param particle:
    :param tau:
    :return:
    """
    xp = np.random.normal(particle[2],
                          math.sqrt(1 / tau))
    yp = np.random.normal(particle[3],
                          math.sqrt(1 / tau))
    x = particle[0] + xp
    y = particle[1] + yp
    moved = np.array([x, y, xp, yp])
    return np.reshape(moved, (1, 4))


def transitions(particles, tau):
    """
    Vectorial version of transition
    :param particles:
    :param tau:
    :return:
    """
    dimpartis = particles.shape[1]
    npartis = particles.shape[0]
    movedparticles = np.zeros((1, dimpartis))
    for i in range(0, npartis):
        #print(movedparticles.shape)
        moved = transition(particles[i, :], tau)
        #print(moved.shape)
        movedparticles = np.append(movedparticles, moved, axis=0)
    return movedparticles[1:, :]


def bootstrap_initialization(mprior, stdprior, z1, N, eta):
    """

    :param mprior:
    :param stdprior:
    :param z1:
    :param N:
    :param eta:
    :return:
    """
    x0s = np.random.normal(mprior[0], stdprior[0], N)
    y0s = np.random.normal(mprior[1], stdprior[1], N)
    xp0s = np.random.normal(mprior[2], stdprior[2], N)
    yp0s = np.random.normal(mprior[3], stdprior[3], N)
    x1s = x0s + xp0s
    y1s = y0s + yp0s
    x1s = x1s.reshape((N, 1))
    y1s = y1s .reshape((N, 1))
    xp0s = xp0s.reshape((N, 1))
    yp0s = yp0s.reshape((N, 1))
    particles = np.concatenate((x1s, y1s, xp0s, yp0s), axis=1)
    newweights = all_logweights(particles, z1, eta)
    normalizedweights = probas.norm_exp_logweights(newweights)
    return particles, normalizedweights


def bootstrap_iteration(previouspartis,
                        z,
                        previousw,
                        tau,
                        eta,
                        res="stratified"):
    if res == "stratified":
        resampled = resampling.stratified_resampling(previouspartis, previousw)
    else :
        resampled = resampling.multi_resampling(previouspartis, previousw)
    moved = transitions(resampled, tau)
    newweights = all_logweights(moved, z, eta)
    normalizedweights = probas.norm_exp_logweights(newweights)
    return moved, normalizedweights


def bootstrap_filter(mprior, stdprior, zs, N, tau, eta,
                     res="stratified"):
    particleslist = []
    weightslist = []
    particles, weights = bootstrap_initialization(mprior,
                                                  stdprior,
                                                  zs[0],
                                                  N,
                                                  eta)
    particleslist.append(particles)
    weightslist.append(weights)
    niter = zs.shape[0]
    for i in range(0, niter):
        particles, weights = bootstrap_iteration(particles,
                                                 zs[i],
                                                 weights,
                                                 tau,
                                                 eta,
                                                 res)
        particleslist.append(particles)
        weightslist.append(weights)
        print(str(i) + "-th iteration")
    return particleslist, weightslist
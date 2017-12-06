import numpy as np
import math
import PFUtils as pfutils
import DataGenerator as datagenerator
import importlib


def gaussian_density(x, m, sigma):
    normcoef = 1/(math.sqrt(2*math.pi)*sigma)
    return normcoef*math.exp(-(x-m)*(x-m)/(2*sigma))


def compute_weight(particle, z, eta):
    m = math.atan(particle[1]/particle[0])
    w = gaussian_density(z, m, eta)
    print(w)
    return w


def compute_weights(particles, z, eta):
    npartis = particles.shape[0]
    w = np.zeros((npartis, ))
    for i in range(0, npartis):
        w[i] = compute_weight(particles[i, :], z, eta)
    return w


def transition(particle, tau):
    """
    Move a particle according to transition kernel
    :param particle:
    :param tau:
    :return:
    """
    xp = np.random.normal(particle[2], 1/tau)
    yp = np.random.normal(particle[3], 1/tau)
    x = particle[0] + xp
    y = particle[1] + yp
    moved = np.array([x, y, xp, yp])
    print(np.reshape(moved, (1, 4)))
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


def bootstrap_initialization(mprior, stdprior, z0, N, eta):
    x0s = np.random.normal(mprior[0], stdprior[0], N)
    y0s = np.random.normal(mprior[1], stdprior[1], N)
    xp0s = np.random.normal(mprior[2], stdprior[2], N)
    yp0s = np.random.normal(mprior[3], stdprior[3], N)
    x0s = x0s.reshape((N, 1))
    y0s = y0s.reshape((N, 1))
    xp0s = xp0s.reshape((N, 1))
    yp0s = yp0s.reshape((N, 1))
    particles = np.concatenate((x0s, y0s, xp0s, yp0s), axis=1)
    newweights = compute_weights(particles, eta, z0)
    normalizedweights = pfutils.normalize_weights(newweights)
    return particles, normalizedweights


def bootstrap_iteration(previouspartis, z, previousw, eta, tau):
    resampled = pfutils.multi_resampling(previouspartis, previousw)
    moved = transitions(resampled, tau)
    newweights = compute_weights(moved, eta, z)
    normalizedweights = pfutils.normalize_weights(newweights)
    return moved, normalizedweights


def bootstrap_filter(mprior, stdprior, zs, N, eta, tau):
    particleslist = []
    weightslist = []
    particles, weights = bootstrap_initialization(mprior, stdprior, zs[0], N, eta)
    particleslist.append(particles)
    weightslist.append(weights)
    niter = zs.shape[0]
    for i in range(0, niter):
        particles, weights = bootstrap_iteration(particles, zs[i], weights, eta, tau)
        particleslist.append(particles)
        weightslist.append(weights)
    return particleslist, weightslist






data = datagenerator.trajectory_data(0.01, 0.95, 0.002, -0.013, 300)
mprior = [0.0108, 1.03, 0.002, -0.013]
stdprior = [0.04, 0.4, 0.003, 0.003]
zs = data["z"].as_matrix()
initp, initw = bootstrap_initialization(mprior, stdprior, zs[0], 400, 0.005)
p, w = bootstrap_iteration(initp, zs[1], initw, 0.005, 1000)
allparticles, allweights = bootstrap_filter(mprior, stdprior, zs, 400, 0.005, 1000)


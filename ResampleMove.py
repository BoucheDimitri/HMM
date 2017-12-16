import numpy as np
import matplotlib.pyplot as plt
import math

import Probas as probas
import Resampling as resampling
import DataGenerator as datagenerator
import AlgebraTools as algtools
import Moves as moves


def augment_one_particle(particle, tau):
    x, y = algtools.particle_to_xyvecs(particle)
    xpaug = np.random.normal(x[-1], math.sqrt(1 / tau))
    ypaug = np.random.normal(y[-1], math.sqrt(1 / tau))
    x = np.append(x, xpaug)
    y = np.append(y, ypaug)
    return np.concatenate((x, y))


def augment_all_particles(particles, tau):
    npartis = particles.shape[0]
    ndims = particles.shape[1]
    augmented = np.zeros((npartis, ndims + 2))
    for i in range(0, npartis):
        augmented[i, :] = augment_one_particle(particles[i, :], tau)
    return augmented


def init_locs(locmean, locstd, N):
    x0s = np.random.normal(locmean[0], locstd[0], (N, 1))
    y0s = np.random.normal(locmean[1], locstd[1], (N, 1))
    return np.concatenate((x0s, y0s), axis=1)


def init_speeds(initloc, speedmean, speedstd, N):
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
                            r1=0.9,
                            restype="stratified"):
    augmented = augment_all_particles(previouspartis, tau)
    t = augmented.shape[1]//2 - 1
    lw = probas.all_logweights(augmented, zs[t], tau, eta)
    normw = probas.norm_exp_logweights(lw)
    if restype == "stratified":
        resampled = resampling.stratified_resampling(augmented, normw, N)
    else:
        resampled = resampling.multi_resampling(augmented, normw, N)
    rescaled = moves.rescale_all_particles(r1, resampled, zs, tau, eta)
    return rescaled, normw
    #return resampled, normw


def resample_move(locmean,
                  locstd,
                  speedmean,
                  speedstd,
                  zs,
                  N,
                  tau,
                  eta,
                  r1,
                  restype):
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
            restype)
        allparticles.append(newparticles)
        allweights.append(weights)
        del particles1
        particles1 = newparticles
        print(t)
    return allparticles, allweights


def extract_loc_means(allparticles):
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











#eta = std of noise in measurement
eta = 0.005
#tau is such that math.sqrt(1/tau) is the std for speeds
tau = 1e6
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
mux = 0
muy = 0
locpriormean = [x0 + mux, y0 + muy]
locpriorstd = [0.04, 0.4]
speedpriormean = [xp0, yp0]
speedpriorstd = [0.001, 0.001]
data = datagenerator.loc_data(x0, y0, xp0, yp0, T, tau, eta)
zs = data["z"].as_matrix()

allparticles, allweights = resample_move(locpriormean,
                                         locpriorstd,
                                         speedpriormean,
                                         speedpriorstd,
                                         zs,
                                         N,
                                         tau,
                                         eta,
                                         0.999,
                                         "stratified")

plt.figure()
plt.plot(data["x"], data["y"], marker="o", label="True trajectory")
means = extract_loc_means(allparticles)
plt.plot(means[0], means[1], marker="o", label="Particle means")
varw = [np.var(w) for w in allweights]

plt.legend()

x, y = particle_to_xyvecs(allparticles[20][0, :])
xps = x[1:]
xc = x_candidate(xps, 15, 20, 0.01)


particle1 = allparticles[10][1, :]
particle2 = allparticles[10][2, :]

aa = rescale_one_particle(0.75, particle1, zs, tau, eta)

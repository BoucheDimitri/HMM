import numpy as np
import math
import PFUtils as pfutils
import DataGenerator as datagenerator
import importlib
import matplotlib.pyplot as plt
import math


def logprop_gauss_ppf(x, m, sigma):
    return -(x-m)*(x-m)/(2*sigma*sigma)


def logprop_pz(z, y, x, eta):
    m = math.atan(y/x)
    return logprop_gauss_ppf(z, m, eta)


def c_matrix(t):
    c = np.zeros((t, t))
    np.fill_diagonal(c, 2*np.ones((t, )))
    c[0, 0] = 1
    c[t-1, t-1] = 1
    diaginf = np.eye(t-1)
    diaginf = np.insert(diaginf, t-1, 0, axis=1)
    diaginf = np.insert(diaginf, 0, 0, axis=0)
    diagsup = np.eye(t-1)
    diagsup = np.insert(diagsup, 0, 0, axis=1)
    diagsup = np.insert(diagsup, t-1, 0, axis=0)
    c -= diaginf + diagsup
    return c


def logprop_prior_ppf(xps, yps, tau):
    k = xps.shape[0]
    cxps = xps.reshape((k, 1))
    cyps = yps.reshape((k, 1))
    cmat = c_matrix(k)
    xterm = -0.5 * tau * np.dot(np.dot(cxps.T,
                                       cmat),
                                cxps)
    yterm = -0.5 * tau * np.dot(np.dot(cyps.T,
                                       cmat),
                                cyps)
    return xterm + yterm


def particle_to_xyvecs(particle):
    """
    Particle must be in order : (x0, xp0, ..., xpk, y0, yp0, ..., ypk)
    :param particle:
    :return:
    """
    x = particle[ :particle.shape[0] // 2].copy()
    y = particle[particle.shape[0] // 2: ].copy()
    return x, y


def one_logweight(particle, zk, tau, eta):
    k = particle.shape[0] - 4
    x, y = particle_to_xyvecs(particle)
    xps = x[1:]
    yps = y[1:]
    pikminus1 = logprop_prior_ppf(xps[:k-1],
                                  yps[:k-1],
                                  tau)
    pik = logprop_prior_ppf(xps[:k],
                            yps[:k],
                            tau)
    xk = np.sum(x)
    yk = np.sum(y)
    lpzk = logprop_pz(zk, yk, xk, eta)
    lpxpk = logprop_gauss_ppf(xps[-1],
                             xps[-2],
                             math.sqrt(1 / tau))
    lpypk = logprop_gauss_ppf(yps[-1],
                             yps[-2],
                             math.sqrt(1 / tau))
    return np.float(pik - pikminus1 + lpzk - lpxpk - lpypk)


def all_logweights(particles, zk, tau, eta):
    npartis = particles.shape[0]
    lw = np.zeros((npartis, ))
    for i in range(0, npartis):
        lw[i] = one_logweight(particles[i, :], zk, tau, eta)
    return lw


def augment_one_particle(particle, tau):
    x, y = particle_to_xyvecs(particle)
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


def initialization(mprior, stdprior, N):
    x0s = np.random.normal(mprior[0], stdprior[0], (N, 1))
    y0s = np.random.normal(mprior[1], stdprior[1], (N, 1))
    xp0s = np.random.normal(mprior[2], stdprior[2], (N, 1))
    yp0s = np.random.normal(mprior[3], stdprior[3], (N, 1))
    particles = np.concatenate((x0s, xp0s, y0s, yp0s), axis=1)
    return particles


def resample_move_iteration(previouspartis,
                            z,
                            tau,
                            eta,
                            N=None,
                            resampling="stratified"):
    augmented = augment_all_particles(previouspartis, tau)
    lw = all_logweights(augmented, z, tau, eta)
    normw = pfutils.norm_exp_logweights(lw)
    if resampling == "stratified":
        resampled = pfutils.stratified_resampling(augmented, normw, N)
    else:
        resampled = pfutils.multi_resampling(augmented, normw, N)
    return resampled, normw


def resample_move(mprior,
                  stdprior,
                  zs,
                  N,
                  tau,
                  eta,
                  deltaN,
                  resampling):
    particles = initialization(mprior, stdprior, N)
    T = zs.shape[0]
    allparticles = []
    allweights = []
    allparticles.append(particles)
    for t in range(1, T):
        particles, weights = resample_move_iteration(
            particles,
            zs[t],
            tau,
            eta,
            N + t*deltaN,
            resampling)
        allparticles.append(particles)
        allweights.append(weights)
        print(t)
    return allparticles, allweights


def extract_loc_means(allparticles):
    locmeansx = []
    locmeansy = []
    for particle in allparticles:
        meanparticle = np.mean(particle, axis=0)
        x, y = particle_to_xyvecs(meanparticle)
        xloc = x[0] + np.sum(x[1:])
        yloc = y[0] + np.sum(y[1:])
        locmeansx.append(xloc)
        locmeansy.append(yloc)
    return locmeansx, locmeansy











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
mux = 0
muy = 0
mprior = [x0 + mux, y0 + muy, xp0, yp0]
stdprior = [0.04, 0.4, 0.001, 0.001]
data = datagenerator.loc_data(x0, y0, xp0, yp0, T, tau, eta)
zs = data["z"].as_matrix()

allparticles, allweights = resample_move(mprior, stdprior, zs, N, tau, eta, 0, "multinomial")


plt.plot(data["x"], data["y"], marker="o", label="True trajectory")

means = extract_loc_means(allparticles)
plt.plot(means[0], means[1], marker="o", label="Particle means")

plt.legend()






toyparticle = np.array([data.loc[0, "x"]])
toyparticle = np.append(toyparticle, np.array([data.loc[1:20, "xp"]]))
toyparticle = np.append(toyparticle, np.array([data.loc[0, "y"]]))
toyparticle = np.append(toyparticle, np.array([data.loc[1:20, "yp"]]))

lw = one_logweight(toyparticle, data.loc[20, "z"], tau, eta)

dd = augment_one_particle(toyparticle, tau)

x, y = particle_to_xyvecs(toyparticle)

toyparticles = np.transpose(np.repeat(
    toyparticle.reshape(
        (toyparticle.shape[0], 1)),
    100,
    axis=1))

ddd = augment_all_particles(toyparticles, tau)

zs = data["z"].as_matrix()

p, w = initialization(mprior, stdprior, zs[0], 1000, 0.005)

nw = pfutils.norm_exp_logweights(w)
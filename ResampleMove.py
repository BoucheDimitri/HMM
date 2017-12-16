import numpy as np
import math
import PFUtils as pfutils
import DataGenerator as datagenerator
import importlib
import matplotlib.pyplot as plt
import math


def particle_to_xyvecs(particle):
    """
    Particle must be in order : (x0, xp0, ..., xpk, y0, yp0, ..., ypk)
    :param particle: numupy.array, a particle
    :return: tuple (particle containing x_0, xp_0, ..., xp_t,
                    particle containing y_0, yp_0, ..., yp_t)
    """
    x = particle[ :particle.shape[0] // 2].copy()
    y = particle[particle.shape[0] // 2: ].copy()
    return x, y


def logprop_gauss_ppf(x, m, sigma):
    """
    log of gauss density up to an additive constant
    :param x:
    :param m:
    :param sigma:
    :return:
    """
    return -(x-m)*(x-m)/(2*sigma*sigma)


def logprop_pz(z, y, x, eta):
    """
    compute log(p(z|x, y)) up to an additive constant
    :param z: float, bearing
    :param y: float, yloc
    :param x: float, xloc
    :param eta: float, measurement error's std
    :return: float, log(p(z|x, y))
    """
    m = math.atan(y/x)
    return logprop_gauss_ppf(z, m, eta)


def logprop_pz_traj(particle, zs, eta):
    """
    compute log(p(z_0,...,z_t|a particle at time t))
    up to an additive constant
    :param particle: numpy.array, the particle at time t
    :param zs: numpy.array, the bearings (at least up to time t)
    :param eta: float, measurement error's std
    :return: float, log(p(z_0,...,z_t|a particle at time t))
    up to an additive constant
    """
    x, y = particle_to_xyvecs(particle)
    xlocs = np.cumsum(x)
    ylocs = np.cumsum(y)
    t = xlocs.shape[0]
    lp = 0
    for i in range(0, t):
        lp += logprop_pz(zs[i], ylocs[i], xlocs[i], eta)
    return lp


def c_matrix(t):
    """
    Generate the C matrix from the article (covariance matrix
    of the gaussian prior on speed vectors)
    :param t: dimension of c matrix
    :return: numpy array, c matrix
    """
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


def logprop_prior_speed(xps, yps, tau):
    """
    Our gaussian prior on a speed vector
    reflecting our prior on the smoothness of
    the trajectory.
    In log and up to an additive constant
    :param xps:
    :param yps:
    :param tau:
    :return:
    """
    k = xps.shape[0]
    cxps = xps.copy().reshape((k, 1))
    cyps = yps.copy().reshape((k, 1))
    cmat = c_matrix(k)
    xterm = -0.5 * tau * np.dot(np.dot(cxps.T,
                                       cmat),
                                cxps)
    yterm = -0.5 * tau * np.dot(np.dot(cyps.T,
                                       cmat),
                                cyps)
    return xterm + yterm


def one_logweight(particle, zk, tau, eta):
    k = particle.shape[0] - 4
    x, y = particle_to_xyvecs(particle)
    xps = x[1:]
    yps = y[1:]
    pikminus1 = logprop_prior_speed(xps[:k-1],
                                  yps[:k-1],
                                  tau)
    pik = logprop_prior_speed(xps[:k],
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
    #return np.float(pik - pikminus1 + lpzk - lpxpk - lpypk)
    return np.float(lpzk)


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


def lkl_ratio(particle1, particle2, zs, tau, eta):
    x1, y1 = particle_to_xyvecs(particle1)
    x2, y2 = particle_to_xyvecs(particle2)
    xp1, yp1 = x1[1:], y1[1:]
    xp2, yp2 = x2[1:], y2[1:]
    lpz1 = logprop_pz_traj(particle1, zs, eta)
    lpz2 = logprop_pz_traj(particle2, zs, eta)
    logprior1 = logprop_prior_speed(xp1, yp1, tau)
    logprior2 = logprop_prior_speed(xp2, yp2, tau)
    return np.exp(np.float(lpz1 + logprior1 - lpz2 - logprior2))


def rescale_one_particle(r1, particle, zs, tau, eta):
    lamb = np.random.uniform(r1, 1/r1)
    rescaled = lamb*particle.copy()
    u = np.random.uniform()
    lklratio = lkl_ratio(rescaled, particle, zs, tau, eta)
    arprob = min(lklratio, 1)
    if u < arprob:
        return rescaled
    else:
        return particle


def rescale_all_particles(r1, particles, zs, tau, eta):
    for i in range(0, particles.shape[0]):
        rescaled = rescale_one_particle(r1, particles[i, :], zs, tau, eta)
        particles[i, :] = rescaled
    return particles


def initialization(mprior, stdprior, N):
    x0s = np.random.normal(mprior[0], stdprior[0], (N, 1))
    y0s = np.random.normal(mprior[1], stdprior[1], (N, 1))
    xp0s = np.random.normal(mprior[2], stdprior[2], (N, 1))
    yp0s = np.random.normal(mprior[3], stdprior[3], (N, 1))
    particles = np.concatenate((x0s, xp0s, y0s, yp0s), axis=1)
    return particles


def resample_move_iteration(previouspartis,
                            zs,
                            tau,
                            eta,
                            r1=0.9,
                            N=None,
                            resampling="stratified"):
    augmented = augment_all_particles(previouspartis, tau)
    t = augmented.shape[1]//2
    lw = all_logweights(augmented, zs[t], tau, eta)
    normw = pfutils.norm_exp_logweights(lw)
    if resampling == "stratified":
        resampled = pfutils.stratified_resampling(augmented, normw, N)
    else:
        resampled = pfutils.multi_resampling(augmented, normw, N)
    rescaled = rescale_all_particles(r1, resampled, zs, tau, eta)
    return resampled, normw


def resample_move(mprior,
                  stdprior,
                  zs,
                  N,
                  tau,
                  eta,
                  r1,
                  deltaN,
                  resampling):
    particles = initialization(mprior, stdprior, N)
    T = zs.shape[0]
    allparticles = []
    allweights = []
    allparticles.append(particles)
    for t in range(1, T-2):
        newparticles, weights = resample_move_iteration(
            particles,
            zs,
            tau,
            eta,
            r1,
            N + t*deltaN,
            resampling)
        allparticles.append(newparticles)
        allweights.append(weights)
        del particles
        particles = newparticles
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
mprior = [x0 + mux, y0 + muy, xp0, yp0]
stdprior = [0.04, 0.4, 0.001, 0.001]
data = datagenerator.loc_data(x0, y0, xp0, yp0, T, tau, eta)
zs = data["z"].as_matrix()

allparticles, allweights = resample_move(mprior, stdprior, zs, N, tau, eta, 1, 0, "stratified")

plt.figure()
plt.plot(data["x"], data["y"], marker="o", label="True trajectory")
means = extract_loc_means(allparticles)
plt.plot(means[0], means[1], marker="o", label="Particle means")
varw = [np.var(w) for w in allweights]

plt.legend()

particle1 = allparticles[10][1, :]
particle2 = allparticles[10][2, :]

aa = rescale_one_particle(0.75, particle1, zs, tau, eta)

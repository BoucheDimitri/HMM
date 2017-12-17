import numpy as np
import math

import AlgebraTools as algtools


def norm_exp_logweights(lw):
    """
    Compute a quantity propotionnal to the weights
    up to the multiplicative constant exp(max(lw))
    which does not matter since it simplifies out
    in the normalization, but it limits rounding errors
    which may occurs since the weights are small quantities
    :param lw: np.array, array of logweights
    :return:
    """
    w = np.exp(lw-np.max(lw))
    return w/np.sum(w)


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


def logprop_gauss_ppf(x, m, sigma):
    """
    log of gauss density up to an additive constant
    :param x: float, x
    :param m: float, mean
    :param sigma: float, std
    :return: float, log proportionnal to log of gaussian density
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
    x, y = algtools.particle_to_xyvecs(particle)
    xlocs = np.cumsum(x)
    ylocs = np.cumsum(y)
    t = xlocs.shape[0]
    lp = 0
    for i in range(0, t):
        lp += logprop_pz(zs[i], ylocs[i], xlocs[i], eta)
    return lp


def logprop_prior_speed(xps, yps, tau):
    """
    Our gaussian prior on a speed vector
    reflecting our prior on the smoothness of
    the trajectory.
    In log and up to an additive constant
    :param xps: vector of speeds in x
    :param yps: vector of speeds in y
    :param tau: inverse of variance of speeds
    :return: prior on a speed vector, in log up to an additive constant
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
    """
    Compute logweights up to an additive constant for a given particle
    Some lines are now useless, they were used to compute
    the complex theoretical weights from the article
    which turned out to work not so greatly
    :param particle: numpy array, particle
    :param zk: float, bearing observation
    :param tau: inverse of variance of speeds
    :param eta: std of noise on bearings measurement
    :return: log of weight for the particle up to an additive constant
    """
    k = particle.shape[0] - 4
    x, y = algtools.particle_to_xyvecs(particle)
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
    """
    Compute weights as done on one particle in
    one_logweight for all particles
    :param particles: numpy array, array of particles
    :param zk: float, bearing observation
    :param tau: inverse of variance of speeds
    :param eta: std of noise on bearings measurement
    :return: numpy array,
    the array of log of weight for the particle up to an additive constant
    """
    npartis = particles.shape[0]
    lw = np.zeros((npartis, ))
    for i in range(0, npartis):
        lw[i] = one_logweight(particles[i, :], zk, tau, eta)
    return lw


def lkl_ratio(particle1, particle2, zs, tau, eta):
    """
    compute the likelyhood ratio given the data zs
    for two particles. Will be used for the Metropolis
    Hastings moves
    :param particle1: numpy array, a particle
    :param particle2:  numpy array, a particle
    :param zs: numpy array, bearings data
    :param tau: inverse of variance of speeds
    :param eta: std of noise on bearings measurement
    :return:
    """
    x1, y1 = algtools.particle_to_xyvecs(particle1)
    x2, y2 = algtools.particle_to_xyvecs(particle2)
    xp1, yp1 = x1[1:], y1[1:]
    xp2, yp2 = x2[1:], y2[1:]
    lpz1 = logprop_pz_traj(particle1, zs, eta)
    lpz2 = logprop_pz_traj(particle2, zs, eta)
    logprior1 = logprop_prior_speed(xp1, yp1, tau)
    logprior2 = logprop_prior_speed(xp2, yp2, tau)
    return np.exp(np.float(lpz1 + logprior1 - lpz2 - logprior2))

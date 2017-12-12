import numpy as np
import math
import PFUtils as pfutils
import DataGenerator as datagenerator
import importlib
import matplotlib.pyplot as plt
import math


def logprop_gauss_ppf(x, m, sigma):
    return -math.log(sigma)-(x-m)*(x-m)/(2*sigma*sigma)


def logprop_pz(z, y, x, eta):
    m = math.atan(y/x)
    return logprop_gauss_ppf(z, m, eta)


def c_matrix(t):
    c = np.zeros((t+1, t+1))
    np.fill_diagonal(c, 2*np.ones((t+1, )))
    c[0, 0] = 1
    c[t, t] = 1
    diaginf = np.eye(t)
    diaginf = np.insert(diaginf, t, 0, axis=1)
    diaginf = np.insert(diaginf, 0, 0, axis=0)
    diagsup = np.eye(t)
    diagsup = np.insert(diagsup, 0, 0, axis=1)
    diagsup = np.insert(diagsup, t, 0, axis=0)
    c -= diaginf + diagsup
    return c


def logprop_prior_ppf(xps, yps, tau):
    k = xps.shape[0]
    cxps = xps.reshape((k + 1, 1))
    cyps = yps.reshape((k + 1, 1))
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
    x = particle[ :particle.shape[0] // 2]
    y = particle[particle.shape[0] // 2: ]
    return x, y


def one_logweight(particle, zk, tau, eta):
    k = particle.shape[0] - 4
    x, y = particle_to_xyvecs(particle)
    xps = x[1:]
    yps = y[1:]
    pikminus1 = logprop_prior_ppf(xps[:, k-1],
                                  yps[:, k-1],
                                  tau)
    pik = logprop_prior_ppf(xps[:, k],
                            yps[:, k],
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
    return pik - pikminus1 + lpzk - lpxpk - lpypk


def initialization(mprior, stdprior, N):
    x0s = np.random.normal(mprior[0], stdprior[0], N)
    y0s = np.random.normal(mprior[1], stdprior[1], N)
    xp0s = np.random.normal(mprior[2], stdprior[2], N)
    yp0s = np.random.normal(mprior[3], stdprior[3], N)

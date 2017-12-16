import numpy as np

import Probas as probas


def rescale_one_particle(r1, particle, zs, tau, eta):
    lamb = np.random.uniform(r1, 1/r1)
    rescaled = particle.copy()
    rescaled *= lamb
    u = np.random.uniform()
    lklratio = probas.lkl_ratio(rescaled, particle, zs, tau, eta)
    arprob = min(lklratio, 1)
    if u < arprob:
        return rescaled
    else:
        return particle.copy()


def rescale_all_particles(r1, particles, zs, tau, eta):
    for i in range(0, particles.shape[0]):
        rescaled = rescale_one_particle(r1, particles[i, :], zs, tau, eta)
        particles[i, :] = rescaled
    return particles

def c_restriction(t, a, b):
    c = probas.c_matrix(t)
    crestr = c[a:b, a:b]
    return crestr


def x_candidate(xps, a, b, r2):
    candidate = xps.copy()
    t = xps.shape[0]
    cres = c_restriction(t, a, b)
    perturbation = np.random.multivariate_normal(candidate[a:b],
                                                 r2*np.linalg.inv(cres))
    candidate[a:b] = perturbation
    return candidate


def y_candidate(xcandidate, a, b):
    xlocs = np.cumsum(xcandidate)

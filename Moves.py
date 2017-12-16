import numpy as np

import Probas as probas
import AlgebraTools as algtools


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


def y_candidate(xlocs, ypinit, a, b, zs, eta):
    candidate = ypinit.copy()
    tanmat = algtools.tan_matrix(a, b, zs)
    mu = np.dot(tanmat, xlocs[a:b+1])
    covmat = algtools.eta_matrix(b-a, eta)
    perturbation = np.random.multivariate_normal(mu, covmat)
    candidate[a:b] = perturbation
    return candidate


def perturbed_particle(particle, zs, a, b, r2, eta):
    x, y = algtools.particle_to_xyvecs(particle)
    xps = x[1:]
    yps = y[1:]
    xpcandidate = x_candidate(xps, a, b, r2)
    xcandidate = x.copy()
    xcandidate[1:] = xpcandidate
    xlocs = np.cumsum(xcandidate)
    ypcandidate = y_candidate(xlocs, yps, a, b, zs, eta)
    ycandidate = y.copy()
    ycandidate[1:] = ypcandidate
    return np.concatenate((xcandidate, ycandidate))


def perturb_one_particle(particle, zs, a, b, r2, tau, eta):
    perturbed = perturbed_particle(particle, zs, a, b, r2, eta)
    mhratio = probas.lkl_ratio(perturbed, particle, zs, tau, eta)
    arprob = min(1, mhratio)
    u = np.random.uniform()
    print(arprob)
    if u < arprob:
        return perturbed
    else:
        return particle


def pertub_all_particles(particles, zs, a, b, r2, tau, eta):
    for i in range(0, particles.shape[0]):
        perturbed = perturb_one_particle(particles[i, :], zs, a, b, r2, tau, eta)
        particles[i, :] = perturbed
    return particles









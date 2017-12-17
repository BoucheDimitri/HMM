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


def c_restriction(t, a, b, c, d):
    cmat = probas.c_matrix(t)
    crestr = cmat[a:b, c:d]
    return crestr


def mu_inde_proposal(xps, a, b):
    t = xps.shape[0]
    cab = c_restriction(t, a, b, a, b)
    print(cab)
    cant = c_restriction(t, a, b, 0, a-1)
    print(cant)
    if t-b-1 <= 0:
        term2 = 0
    else:
        cfut = c_restriction(t, a, b, b+1, t)
        print(cfut)
        term2 = np.dot(cfut,
                       xps[b + 1:].reshape((t - b - 1, 1)))
    term1 = np.dot(cant,
                   xps[:a - 1].reshape((a-1, 1)))
    cinv = np.linalg.inv(cab)
    muab = np.dot(cinv, term1 + term2)
    return muab


def x_candidate(xps, a, b, r2):
    candidate = xps.copy()
    t = xps.shape[0]
    cres = c_restriction(t, a, b, a, b)
    perturbation = np.random.multivariate_normal(candidate[a:b],
                                                 r2*np.linalg.inv(cres))
    candidate[a:b] = perturbation
    return candidate


def y_candidate(xlocs, ypinit, a, b, zs, eta):
    candidate = ypinit.copy()
    tanmat = algtools.tan_matrix(a, b, zs)
    #print(tanmat)
    mu = np.dot(tanmat, xlocs[a:b+1])
    #print(mu)
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
    #return np.concatenate((xcandidate, ycandidate))
    return np.concatenate((xcandidate, y.copy()))


def perturbed_particle_bis(particle, a, b, r2, r3):
    x, y = algtools.particle_to_xyvecs(particle)
    xps = x[1:]
    yps = y[1:]
    xpcandidate = x_candidate(xps, a, b, r2)
    xcandidate = x.copy()
    xcandidate[1:] = xpcandidate
    ypcandidate = x_candidate(yps, a, b, r3)
    ycandidate = y.copy()
    ycandidate[1:] = ypcandidate
    return np.concatenate((xcandidate, ycandidate))
    #return np.concatenate((xcandidate, y.copy()))


def perturb_one_particle(particle, zs, a, b, r2, tau, eta):
    perturbed = perturbed_particle(particle, zs, a, b, r2, eta)
    mhratio = probas.lkl_ratio(perturbed, particle, zs, tau, eta)
    arprob = min(1, mhratio)
    u = np.random.uniform()
    #print(arprob)
    if u < arprob:
        return perturbed
    else:
        return particle


def perturb_one_particle_bis(particle, zs, a, b, r2, r3, tau, eta):
    perturbed = perturbed_particle_bis(particle, a, b, r2, r3)
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


def pertub_all_particles_bis(particles, zs, a, b, r2, r3, tau, eta):
    for i in range(0, particles.shape[0]):
        perturbed = perturb_one_particle_bis(particles[i, :], zs, a, b, r2, r3, tau, eta)
        particles[i, :] = perturbed
    return particles










import numpy as np

import Probas as probas
import AlgebraTools as algtools

def perturb_one_tau(particle, d0, c0):
    """
    return one moved tau for one particle
        Particle must be in order : (x0, xp0, ..., xpk, y0, yp0, ..., ypk)
    :param particle: numpy.array, a particle
    :param d0: scalar
    :param c0: scalar
    """
    x, y = algtools.particle_to_xyvecs(particle)
    xps = x[2:]
    xps_shift = x[1:len(x)-1]
    yps = y[2:]
    yps_shift = y[1:len(y)-1]
    t = len(x)-1
    d = d0+2*t
    c = c0+sum((xps-xps_shift)**2)+sum((yps-yps_shift)**2)
    perturbed = np.random.gamma(shape=d, scale=c)
    return(perturbed)
    
def pertub_all_tau(particles, d0, c0):
    tau_moved = np.zeros(particles.shape[0])
    for i in range(0, particles.shape[0]):
        perturbed = perturb_one_tau(particles[i, :], d0, c0)
        tau_moved[i] = perturbed
    return tau_moved

#test perturb_one_tau
particle = np.random.normal(size=10)
tau = 10**6
d0 = 1
c0 = tau
perturb_one_tau(particle, d0, c0)

#test perturb_all_tau
#pertub_all_tau(particles, d0, c0)



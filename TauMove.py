import numpy as np

import AlgebraTools as algtools
from scipy.stats import invgamma

def perturb_one_tau(particle, d0, c0):
    """
    return one moved tau for one particle
        Particle must be in order : (x0, xp0, ..., xpk, y0, yp0, ..., ypk)
    :param particle: numpy.array, a particle
    :param d0: scalar
    :param c0: scalar
    """
    x, y = algtools.particle_to_xyvecs(particle)
    t = len(x)-1
    if t==1:
        perturbed = 1/invgamma.rvs(d0, scale=c0)
    else:
        xps = np.array(x[2:])
        xps_shift = np.array(x[1:(len(x)-1)])
        yps = np.array(y[2:])
        yps_shift = np.array(y[1:(len(y)-1)])
        d = d0+t-1
        c = c0+sum((xps-xps_shift)**2)/2+sum((yps-yps_shift)**2)/2
        perturbed = 1/invgamma.rvs(d, scale=c)
        
    return(perturbed)
    
def pertub_all_tau(particles, d0, c0):
    tau_moved = np.zeros(particles.shape[0])
    for i in range(0, particles.shape[0]):
        perturbed = perturb_one_tau(particles[i, :], d0, c0)
        tau_moved[i] = perturbed
    return tau_moved

'''
#test perturb_one_tau
particle = np.random.normal(size=10)
tau = 10**6
d0 = 1
c0 = tau
inverse_tau = perturb_one_tau(particle, d0, c0)
#simul_tau = pertub_all_tau(particles1, d0, c0)
np.random.gamma(d0,c0)
invgamma.rvs(d0, scale=c0)
np.mean(simul_tau)
np.var(simul_tau)

#test perturb_all_tau
#pertub_all_tau(particles, d0, c0)
#one_particle = particles1[0,:]
x, y = algtools.particle_to_xyvecs(one_particle)
pertub_all_tau(allparticlesrtm[2], d0, c0)
invgamma.rvs(3, scale=2*tau)
pertub_all_tau(allparticlesrtm[2], d0, c0)
pertub_all_tau(allparticlesrtm[10], d0, c0)
'''
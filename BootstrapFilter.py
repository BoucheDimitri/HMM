import numpy as np
import math
import PFUtils as pfutils
import DataGenerator as datagenerator
import importlib
import matplotlib.pyplot as plt
import math


#Make sure the last version of
#pfutils and datagenerator are used
importlib.reload(datagenerator)
importlib.reload(pfutils)


def logprop_gauss_ppf(x, m, sigma):
    return -(x-m)*(x-m)/(2*sigma*sigma)


def one_logweight(particle, z, eta):
    m = math.atan(particle[1]/particle[0])
    lw = logprop_gauss_ppf(z, m, eta)
    return lw


def all_logweights(particles, z, eta):
    npartis = particles.shape[0]
    lw = np.zeros((npartis, ))
    for i in range(0, npartis):
        lw[i] = one_logweight(particles[i, :], z, eta)
    return lw


def transition(particle, tau):
    """
    Move a particle according to transition kernel
    :param particle:
    :param tau:
    :return:
    """
    xp = np.random.normal(particle[2],
                          math.sqrt(1 / tau))
    yp = np.random.normal(particle[3],
                          math.sqrt(1 / tau))
    x = particle[0] + xp
    y = particle[1] + yp
    moved = np.array([x, y, xp, yp])
    return np.reshape(moved, (1, 4))


def transitions(particles, tau):
    """
    Vectorial version of transition
    :param particles:
    :param tau:
    :return:
    """
    dimpartis = particles.shape[1]
    npartis = particles.shape[0]
    movedparticles = np.zeros((1, dimpartis))
    for i in range(0, npartis):
        #print(movedparticles.shape)
        moved = transition(particles[i, :], tau)
        #print(moved.shape)
        movedparticles = np.append(movedparticles, moved, axis=0)
    return movedparticles[1:, :]


def bootstrap_initialization(mprior, stdprior, z1, N, eta):
    """

    :param mprior:
    :param stdprior:
    :param z1:
    :param N:
    :param eta:
    :return:
    """
    x0s = np.random.normal(mprior[0], stdprior[0], N)
    y0s = np.random.normal(mprior[1], stdprior[1], N)
    xp0s = np.random.normal(mprior[2], stdprior[2], N)
    yp0s = np.random.normal(mprior[3], stdprior[3], N)
    x1s = x0s + xp0s
    y1s = y0s + yp0s
    x1s = x1s.reshape((N, 1))
    y1s = y1s .reshape((N, 1))
    xp0s = xp0s.reshape((N, 1))
    yp0s = yp0s.reshape((N, 1))
    particles = np.concatenate((x1s, y1s, xp0s, yp0s), axis=1)
    newweights = all_logweights(particles, z1, eta)
    normalizedweights = pfutils.norm_exp_logweights(newweights)
    return particles, normalizedweights


def bootstrap_iteration(previouspartis, z, previousw, tau, eta):
    resampled = pfutils.multi_resampling(previouspartis, previousw)
    moved = transitions(resampled, tau)
    newweights = all_logweights(moved, z, eta)
    normalizedweights = pfutils.norm_exp_logweights(newweights)
    return moved, normalizedweights


def bootstrap_filter(mprior, stdprior, zs, N, tau, eta):
    particleslist = []
    weightslist = []
    particles, weights = bootstrap_initialization(mprior, stdprior, zs[0], N, eta)
    particleslist.append(particles)
    weightslist.append(weights)
    niter = zs.shape[0]
    for i in range(0, niter):
        particles, weights = bootstrap_iteration(particles, zs[i], weights, tau, eta)
        particleslist.append(particles)
        weightslist.append(weights)
        print(str(i) + "-th iteration")
    return particleslist, weightslist

#eta = std of noise in measurement
eta = 0.0005
#tau is such that math.sqrt(1/tau) is the std for speeds
tau = 100000
#N is the number of particles
N = 1000
#T is the number of periods
T = 100
#Initial conditions
x0 = 20
y0 = 20
xp0 = 0.002
yp0 = -0.013
mprior = [20.01, 19.6, 0.002, -0.013]
stdprior = [0.04, 0.4, 0.003, 0.003]


data = datagenerator.loc_data(x0, y0, xp0, yp0, T, tau, eta)
#Delete first observation
zs = data["z"].as_matrix()[1:]
#initp, initw = bootstrap_initialization(mprior, stdprior, zs[0], tau, eta)
#p, w = bootstrap_iteration(initp, zs[1], initw, 0.005, 1000)
allparticles, allweights = bootstrap_filter(mprior, stdprior, zs, N, tau, eta)

means = np.array([np.mean(a, axis=0) for a in allparticles])
varw = [np.var(w) for w in allweights]



plt.scatter(means[:, 0], means[:, 1], label="particle_means")
plt.scatter(data["x"][1:], data["y"][1:], label="real_trajectory")
plt.legend()



plt.scatter(data.loc[1, "x"], data.loc[1, "y"])
plt.scatter(data.loc[2, "x"], data.loc[2, "y"])
plt.scatter(data.loc[3, "x"], data.loc[3, "y"])
plt.scatter(data.loc[4, "x"], data.loc[4, "y"])
plt.scatter(allparticles[1][:, 0], allparticles[1][:, 1], s=1, label="1")
plt.scatter(allparticles[2][:, 0], allparticles[2][:, 1], s=1, label="2")
plt.scatter(allparticles[3][:, 0], allparticles[3][:, 1], s=1, label="3")

plt.scatter(data.loc[50, "x"], data.loc[50, "y"])
plt.scatter(allparticles[50][:, 0], allparticles[50][:, 1], s=1, label="50")


plt.scatter(allparticles[100][:, 0], allparticles[100][:, 1], s=1, label="100")
plt.scatter(allparticles[150][:, 0], allparticles[150][:, 1], s=1, label="150")
plt.scatter(allparticles[200][:, 0], allparticles[200][:, 1], s=1, label="200")
plt.legend()

plt.plot(data["x"], data["y"])
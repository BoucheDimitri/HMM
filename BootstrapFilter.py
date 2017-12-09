import numpy as np
import math
import PFUtils as pfutils
import DataGenerator as datagenerator
import importlib
import matplotlib.pyplot as plt


def logprop_gauss_ppf(x, m, sigma):
    return -(x-m)*(x-m)/(2*sigma)


def one_logweight(particle, z, eta):
    m = math.atan(particle[1]/particle[0])
    lw = logprop_gauss_ppf(z, m, eta)
    print(lw)
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
    xp = np.random.normal(particle[2], 1/tau)
    yp = np.random.normal(particle[3], 1/tau)
    x = particle[0] + particle[2]
    y = particle[1] + particle[3]
    moved = np.array([x, y, xp, yp])
    print(np.reshape(moved, (1, 4)))
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
    newweights = all_logweights(particles, eta, z1)
    normalizedweights = pfutils.norm_exp_logweights(newweights)
    return particles, normalizedweights


def bootstrap_iteration(previouspartis, z, previousw, eta, tau):
    resampled = pfutils.multi_resampling(previouspartis, previousw)
    moved = transitions(resampled, tau)
    newweights = all_logweights(moved, eta, z)
    normalizedweights = pfutils.norm_exp_logweights(newweights)
    return moved, normalizedweights


def bootstrap_filter(mprior, stdprior, zs, N, eta, tau):
    particleslist = []
    weightslist = []
    particles, weights = bootstrap_initialization(mprior, stdprior, zs[0], N, eta)
    particleslist.append(particles)
    weightslist.append(weights)
    niter = zs.shape[0]
    for i in range(0, niter):
        particles, weights = bootstrap_iteration(particles, zs[i], weights, eta, tau)
        particleslist.append(particles)
        weightslist.append(weights)
    return particleslist, weightslist






data = datagenerator.loc_data(20, 20, 0.002, -0.013, 100)
mprior = [20, 20, 0.002, -0.013]
stdprior = [0.04, 0.4, 0.003, 0.003]
#Delete first observation
zs = data["z"].as_matrix()[1:]
#initp, initw = bootstrap_initialization(mprior, stdprior, zs[0], 1000, 0.005)
#p, w = bootstrap_iteration(initp, zs[1], initw, 0.005, 1000)
allparticles, allweights = bootstrap_filter(mprior, stdprior, zs, 1000, 0.005, 1000)

means = np.array([np.mean(a, axis=0) for a in allparticles])
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
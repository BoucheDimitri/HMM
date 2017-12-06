import numpy as np


def normalize_weights(weights):
    nweights = np.copy(weights)
    return (1/np.sum(weights))*nweights


def multi_resampling(particles, weights):
    """
    Perform multinomial resampling
    :param particles: np.array with shape=(n_particles, dim_particles)
    :param weights: np.array with shape=(1, nparticles) or list
    :return: np.array with shape=(n_particles, dim_particles) :
    the resampled particles
    """
    npartis = particles.shape[0]
    multi = np.random.multinomial(npartis, weights)
    resampled = np.zeros((1, particles.shape[1]))
    print(resampled)
    loc = 0
    for i in range(0, npartis):
        nrep = multi[i]
        reps = np.tile(particles[i, :], (nrep, 1))
        print(reps.shape)
        print(resampled.shape)
        resampled = np.append(resampled, reps, axis=0)
    return resampled[1:, :]



#Test part for multinomial resampling
#To remove
#particules = np.zeros((10, 4))
#for i in range(0, 10):
#particules[i, :] = i*np.ones((1, 4))
#w = np.array([0.1, 0.3, 0.05, 0.05, 0.3, 0, 0, 0.1, 0.1, 0])
#res = multi_resampling(particules, w)


#Test for weights_normalization
#w = np.array([1, 3, 5, 10, 3, 4, 5])
#wn = normalize_weights(w)
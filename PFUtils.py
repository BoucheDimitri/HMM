import numpy as np
import bisect


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


def n_stratified(weights, N=None):
    if not N:
        N = weights.shape[0]
    uniforms = []
   # Nw = weights.shape[0]
    for n in range(1, N+1):
        uniforms.append(np.random.uniform((n-1)/N, n/N))
    uniforms = np.array(uniforms)
    cumweights = np.cumsum(weights)
    inds = np.zeros((N, 1))
    multi = np.zeros((N, ))
    count = 0
    for u in uniforms:
        index = bisect.bisect(cumweights, u)
        inds[count] = index + 1
        count += 1
    for n in range(1, N+1):
        multi[n-1] = np.argwhere(inds == n).shape[0]
    return [np.int(m) for m in multi]


def n_multinomial(weights, N=None):
    if not N:
        N = weights.shape[0]
    multi = np.random.multinomial(N, weights)
    return multi


def multi_resampling(particles, weights, N=None):
    """
    Perform multinomial resampling
    :param particles: np.array with shape=(n_particles, dim_particles)
    :param weights: np.array with shape=(1, nparticles) or list
    :return: np.array with shape=(n_particles, dim_particles) :
    the resampled particles
    """
    if not N:
        N = particles.shape[0]
    multi = n_multinomial(weights, N)
    resampled = np.zeros((1, particles.shape[1]))
    for i in range(0, multi.shape[0]):
        nrep = multi[i]
        reps = np.tile(particles[i, :], (nrep, 1))
        resampled = np.append(resampled, reps, axis=0)
    return resampled[1:, :]


def stratified_resampling(particles, weights, N=None):
    """
    Perform multinomial resampling
    :param particles: np.array with shape=(n_particles, dim_particles)
    :param weights: np.array with shape=(1, nparticles) or list
    :return: np.array with shape=(n_particles, dim_particles) :
    the resampled particles
    """
    if not N:
        N = particles.shape[0]
    multi = n_stratified(weights, N)
    resampled = np.zeros((1, particles.shape[1]))
    for i in range(0, len(multi)):
        nrep = multi[i]
        reps = np.tile(particles[i, :], (nrep, 1))
        resampled = np.append(resampled, reps, axis=0)
    return resampled[1:, :]











#Test part for multinomial resampling
#To remove
particles = np.zeros((20, 4))
for i in range(0, 20):
    particles[i, :] = i*np.ones((1, 4))
w = np.array([0.25, 0, 0, 0, 0, 0.25, 0, 0, 0, 0, 0.25, 0, 0, 0, 0, 0.25, 0, 0, 0, 0])

#dd = n_multinomial(w, 40)
#res = stratified_resampling(particles, w, 40)

#m = n_stratified(w, 40)
#res = stratified_resampling(particules, w)


#Test for weights_normalization
#w = np.array([1, 3, 5, 10, 3, 4, 5])
#wn = normalize_weights(w)
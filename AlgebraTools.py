import math
import numpy as np


def particle_to_xyvecs(particle):
    """
    Particle must be in order : (x0, xp0, ..., xpk, y0, yp0, ..., ypk)
    :param particle: numupy.array, a particle
    :return: tuple (particle containing x_0, xp_0, ..., xp_t,
                    particle containing y_0, yp_0, ..., yp_t)
    """
    x = particle[ :particle.shape[0] // 2].copy()
    y = particle[particle.shape[0] // 2: ].copy()
    return x, y


def tan_matrix(a, b, zs, noise=0):
    tmat = np.zeros((b-a, b-a+1))
    diag = [-math.tan(zs[a+i] + np.random.normal(0, noise)) for i in range(0, b-a)]
    np.fill_diagonal(tmat, diag)
    fwdmat = np.zeros((b-a, b-a))
    diagbis = [math.tan(zs[a+i+1] + np.random.normal(0, noise)) for i in range(0, b-a)]
    np.fill_diagonal(fwdmat, diagbis)
    fwdmat = np.concatenate((np.zeros((b-a, 1)), fwdmat), axis=1)
    return tmat + fwdmat


def eta_matrix(t, eta):
    mat = np.zeros((t, t))
    np.fill_diagonal(mat, eta*eta)
    return mat


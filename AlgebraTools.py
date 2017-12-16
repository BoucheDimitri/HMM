


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

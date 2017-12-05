import numpy as np
import pandas as pd
import math


def zfunct(z, eta):
    return math.atan(z) + np.random.normal(0, eta*eta)


def trajectory_data(x0,
                    y0,
                    xp0,
                    yp0,
                    T,
                    tau=1000,
                    eta=0.0005):
    """
    Simulate  trajectory data according
    to the article's procedure
    :param x0: initial x coordinate
    :param y0: initial y coordinate
    :param xp0: initiall speed in x
    :param yp0: initial speed in y
    :param T: size of trajectory (n simulations)
    :param tau: tau parameter from article
    :param eta: eta parameter from article
    :return: data as a pandas dataframe
    """
    #Initialize trajectory dataframe
    data = pd.DataFrame(
        columns=["x", "y", "xp", "yp", "z"],
        index=range(0, T),
        dtype=float)
    #Fill first row with initial conditions
    data.set_value(0, "x", x0)
    data.set_value(0, "y", y0)
    data.set_value(0, "xp", xp0)
    data.set_value(0, "yp", yp0)
    #Simulate and fill dataframe
    for t in range(1, T+1):
        xpt = np.random.normal(
            data.get_value(t - 1, "xp"),
            1 / tau)
        ypt = np.random.normal(
            data.get_value(t - 1, "yp"),
            1 / tau)
        xt = data.get_value(t - 1, "x") \
             + data.get_value(t - 1, "xp")
        yt = data.get_value(t - 1, "y") \
             + data.get_value(t - 1, "yp")
        data.set_value(t, "x", xt)
        data.set_value(t, "y", yt)
        data.set_value(t, "xp", xpt)
        data.set_value(t, "yp", ypt)
    #Fill the bearings column (observed process)
    data["z"] = (data["y"]/data["x"]).apply(
        lambda u: zfunct(eta, u))
    return data





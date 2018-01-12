import numpy as np
import pandas as pd
import math


def zfunct(z, eta):
    return math.atan(z) + np.random.normal(0, eta)


def speed_data(xp0,
               yp0,
               T,
               tau=1000):
    """
    Generate a dataframe of speed data
    as described in Berzuini and Zilks (1999)
    :param xp0: float, initial speed in x
    :param yp0: float, initial speed in y
    :param T: int, number of periods
    :param tau: float, inverse of variance of speed
    :return: pandas.core.frame.DataFrame, the data frame of speed data
    """
    speeds = pd.DataFrame(columns=["xp", "yp"], index=range(0, T), dtype=float)
    speeds.set_value(0, "xp", xp0)
    speeds.set_value(0, "yp", yp0)
    for t in range(1, T):
        xpt = np.random.normal(
            speeds.get_value(t - 1, "xp"),
            math.sqrt(1 / tau))
        ypt = np.random.normal(
            speeds.get_value(t - 1, "yp"),
            math.sqrt(1 / tau))
        speeds.set_value(t, "xp", xpt)
        speeds.set_value(t, "yp", ypt)
    return speeds


def loc_from_speed_data(speeds,
                        x0,
                        y0):
    """
    Generate localization from a dataframe of speeds
    and initial positions
    :param speeds: pandas.core.frame.DataFrame, dataframe of speed
    as returned by function speed_data
    :param x0: float, initial position in x
    :param y0: float, initial position in y
    :return: pandas.core.frame.DataFrame, dataframe of speeds and
    positions
    """
    T = speeds.shape[0]
    locs = pd.DataFrame(columns=["x", "y"],
                        index=range(0, T))
    locs = pd.concat((locs, speeds),
                     axis=1)
    locs.set_value(0, "x", x0)
    locs.set_value(0, "y", y0)
    cumxspeeds = locs["xp"].cumsum().shift(1).fillna(0)
    cumyspeeds = locs["yp"].cumsum().shift(1).fillna(0)
    locs["x"] = locs.loc[0, "x"] + cumxspeeds
    locs["y"] = locs.loc[0, "y"] + cumyspeeds
    return locs


def add_bearings(locdata,
                 eta=0.005):
    locdata["z"] = (locdata["y"]
                    / locdata["x"]).apply(
        lambda u: zfunct(u, eta))
    return locdata


def loc_data(x0,
             y0,
             xp0,
             yp0,
             T,
             tau=1000,
             eta=0.005):
    # génération de l'ensemble des vitesses à partir des vitesses initiales            
    speed = speed_data(xp0, yp0, T, tau)
    # génération des position à partir des vitesse et des positions initiales
    locs = loc_from_speed_data(speed, x0, y0)
    # ajout des bearings
    locs = add_bearings(locs, eta)
    # un DataFrame de cinq colonnes (x, y, xp, yp, z) est retournée
    return locs






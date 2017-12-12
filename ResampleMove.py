import numpy as np
import math
import PFUtils as pfutils
import DataGenerator as datagenerator
import importlib
import matplotlib.pyplot as plt
import math


def c_matrix(t):
    c = np.zeros((t+1, t+1))
    np.fill_diagonal(c, 2*np.ones((t+1, )))
    c[0, 0] = 1
    c[t, t] = 1
    diaginf = np.eye(t)
    diaginf = np.insert(diaginf, t, 0, axis=1)
    diaginf = np.insert(diaginf, 0, 0, axis=0)
    diagsup = np.eye(t)
    diagsup = np.insert(diagsup, 0, 0, axis=1)
    diagsup = np.insert(diagsup, t, 0, axis=0)
    c -= diaginf + diagsup
    return c


def prior_ppf(xps, yps):


def initialization(mprior, stdprior, N):
    x0s = np.random.normal(mprior[0], stdprior[0], N)
    y0s = np.random.normal(mprior[1], stdprior[1], N)
    xp0s = np.random.normal(mprior[2], stdprior[2], N)
    yp0s = np.random.normal(mprior[3], stdprior[3], N)



def
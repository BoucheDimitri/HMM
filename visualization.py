from matplotlib import cm
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import acquisition_max as am
import prediction_formulae as pred


def plot_func_1d(
        bounds,
        grid_size,
        func,
        nsub_plots=1,
        axis=None,
        label=None,
        c=None):
    """
    Plot a function in 1d

    Args :
        bounds (tuple) : ((min_d1, min_d2), (max_d1, max_d2))
        grid_size (tuple) : (gridsize_x, gridsize_y)
        func (function) : 1d func to plot
        nsub_plots (int) : if no axis provided, create new figure with nsub_plots subplots
        axis( : axis to plot on if on already existing figure
        label(str) : legend for func plot

    Returs:
        Depends, if axis is provided, returns None, else returns fig, axes
    """
    if isinstance(bounds[0], tuple):
        grid = np.linspace(bounds[0][0], bounds[0][1], grid_size)
    else:
        grid = np.linspace(bounds[0], bounds[1], grid_size)
    y = [func(np.array([grid[i]])) for i in range(0, grid.shape[0])]
    if axis:
        axis.plot(grid, y, label=label, c=c)
    else:
        fig, axes = plt.subplots(nsub_plots, 1, sharex=True)
        axes[0].plot(grid, y, label=label, c=c)
        return fig, axes


def plot_acq_func_1d(xmat,
                     y,
                     Rinv,
                     beta_hat,
                     theta,
                     p,
                     bounds,
                     grid_size,
                     acq_func,
                     axis):
    """
    Plot tool for 1d acquisition function

    Args :
        xmat (numpy.ndarray) : the data points so far, shape = (n, k)
        y (numpy.ndarray) : y, shape=(n, 1)
        Rinv (numpy.ndarray) : inverse of kernel matrix
        beta_hat (float) : estimation of beta
        theta (numpy.ndarray) : vector of theta params, one by dim, shape = (k, )
        p (numpy.ndarray) : powers used to compute the distance, one by dim, shape = (k, )
        bounds (tuple) : (min, max)
        grid_size (tuple) : (gridsize_x, gridsize_y)
        acq_func : Instance of one of the classes in Acquisition_Functions.py file
        axis (matplotlib.axes._subplots.AxesSubplot) : Axis to plot on

    Returs:
        matplotlib.axes._subplots.AxesSubplot. The axis with the new plot

    """
    def acq_plot(xnew):
        return am.complete_acq_func(xmat,
                                    xnew,
                                    y,
                                    Rinv,
                                    beta_hat,
                                    theta,
                                    p,
                                    acq_func)
    plot_func_1d(
        bounds,
        grid_size,
        acq_plot,
        axis=axis,
        label=acq_func.name)
    return axis


def add_points_1d(ax, points_x, points_y, c=None, label=None):
    if label:
        ax.scatter(points_x, points_y, c=c, label=label)
    else:
        ax.scatter(points_x, points_y, c=c)
    return ax


def plot_gp_means_std(xmat,
                      y,
                      Rinv,
                      beta_hat,
                      theta,
                      p,
                      bounds,
                      grid_size,
                      axis):
    xgrid = np.linspace(bounds[0], bounds[1], grid_size)
    gp_means, gp_stds = pred.pred_means_stds(
        xgrid, xmat, y, Rinv, beta_hat, theta, p)
    plus_1std = gp_means + gp_stds
    minus_1std = gp_means - gp_stds
    axis.plot(xgrid, gp_means)
    axis.fill_between(xgrid, plus_1std, minus_1std, alpha=.3)
    return axis


def bayes_opti_plot_1d(xmat,
                       y,
                       Rinv,
                       beta_hat,
                       theta,
                       p,
                       bounds,
                       grid_size,
                       acq_func,
                       objective_func):
    fig, axes = plot_func_1d(
        bounds, grid_size, objective_func, nsub_plots=2, label="Objective")
    axes[0] = add_points_1d(axes[0], xmat, y)
    axes[1] = plot_acq_func_1d(xmat,
                               y,
                               Rinv,
                               beta_hat,
                               theta,
                               p,
                               bounds,
                               grid_size,
                               acq_func,
                               axis=axes[1])
    axes[0] = plot_gp_means_std(xmat,
                                y,
                                Rinv,
                                beta_hat,
                                theta,
                                p,
                                bounds,
                                grid_size,
                                axes[0])
    plt.legend()
    return axes


def mesh_grid(bounds, grid_size):
    """
    Create a meshgrid for 3d plots

    Args :
        bounds (tuple) : ((min_d1, min_d2), (max_d1, max_d2))
        grid_size (tuple) : (gridsize_x, gridsize_y)

    Returns :
        tuple. Tuple of numpy array
    """
    x_axis = np.linspace(bounds[0][0], bounds[0][1], grid_size[0])
    y_axis = np.linspace(bounds[1][0], bounds[1][1], grid_size[1])
    xgrid, ygrid = np.meshgrid(x_axis, y_axis)
    return xgrid, ygrid


def plot_func_2d(bounds, grid_size, func, title=None, plot_type="ColoredSurface", alpha=None, ax=None):
    """
    3d heated colored surface plot of R^2 to R function

    Args :
        bounds (tuple) : ((min_d1, min_d2), (max_d1, max_d2))
        grid_size (tuple) : (gridsize_x, gridsize_y)
        func (function) : the function to plot
        title (str) : the title for the plot

    Returs:
        nonetype. None

    """
    if not ax:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
    xgrid, ygrid = mesh_grid(bounds, grid_size)
    zgrid = np.zeros(shape=xgrid.shape)
    for i in range(0, xgrid.shape[0]):
        for j in range(0, xgrid.shape[1]):
            zgrid[i, j] = func(np.array([xgrid[i, j], ygrid[i, j]]))
    if plot_type == "WireFrame":
        ax.plot_wireframe(xgrid, ygrid, zgrid, alpha=alpha)
    elif plot_type == "ColoredSurface":
        ax.plot_surface(xgrid, ygrid, zgrid, cmap=cm.coolwarm,
                        linewidth=0.1, antialiased=False, alpha=alpha)
    else:
        ax.plot_surface(xgrid, ygrid, zgrid, linewidth=0.1, antialiased=False, alpha=alpha)
    plt.title(title)
    return ax


def add_points_2d(xmat, y, ax=None):
    if not ax:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
    ax.scatter(xmat[:, 0], xmat[:, 1], y[:, 0], marker="o", s=100, c='k')
    return ax


def plot_acq_func_2d(xmat,
                     y,
                     Rinv,
                     beta_hat,
                     theta,
                     p,
                     bounds,
                     grid_size,
                     acq_func):
    """
    3d heated colored surface plot of acquisition function

    Args :
        xmat (numpy.ndarray) : the data points so far, shape = (n, k)
        y (numpy.ndarray) : y, shape=(n, 1)
        Rinv (numpy.ndarray) : inverse of kernel matrix
        beta_hat (float) : estimation of beta
        theta (numpy.ndarray) : vector of theta params, one by dim, shape = (k, )
        p (numpy.ndarray) : powers used to compute the distance, one by dim, shape = (k, )
        bounds (tuple) : ((min_d1, min_d2), (max_d1, max_d2))
        grid_size (tuple) : (gridsize_x, gridsize_y)
        acq_func : Instance of one of the classes in Acquisition_Functions.py file

    Returs:
        nonetype. None

    """
    def acq_plot(xnew):
        return am.complete_acq_func(xmat,
                                    xnew,
                                    y,
                                    Rinv,
                                    beta_hat,
                                    theta,
                                    p,
                                    acq_func)
    title = acq_func.name
    if title == "EI":
        title += "; xi=" + str(acq_func.xi)
    elif title == "LCB":
        title += "; eta=" + str(acq_func.eta)
    plot_func_2d(bounds, grid_size, acq_plot, title=title)

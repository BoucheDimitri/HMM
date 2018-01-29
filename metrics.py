import numpy as np
import cho_inv
import exp_kernel
import test_functions as test_func
import max_likelihood as max_llk
import bayesian_optimization as bayes_opti
import acquisition_max as am
import visualization as viz
import prediction_formulae as pred
import exp_kernel

import scipy
import math
import pandas as pd

def spread_bylines(mat, point):
    """
    mat (numpy.ndarray) : the data points, shape = (n, k)
    point (numpy.ndarray) : true solution of the function, shape(1, k)
    """
    return abs(mat-point)

#test spread_bylines
#mat = np.array(range(0,4)).reshape((2,2))
#point = np.array([1,1]).reshape((1,2))
#spread_bylines(mat, point)
    
def f_precisionp_metric(y, true_y_sol, p):
    """
     Return the number of iterations required before a point is sampled 
     with an objective function value within p*100% of the true solution
    Args:
        y (numpy.ndarray) : y, shape=(n, 1)
        true_min (float) : true optimum of the function
        p (0<float<1) : percentage of precision
    Returns:
        integer: nb of iterations required before a point is sampled 
        with an objective function value within p*100% of the true min 
        (=0 if no such iteration)           
    """
    
    n = y.shape[0]
    spread = spread_bylines(y, true_y_sol)
    
    for i in range(0,n):
        if spread[i]<abs(p*true_y_sol):
            nb_it_precisionp = i+1
            break
        if i==n-1 and spread[i]>abs(p*true_y_sol):
            nb_it_precisionp = 0
    return nb_it_precisionp

#test f_precisionp_metric
#y=np.random.uniform(0,1,2).reshape(2,1)
#true_min=min(y)-0.001
#p=0.000000001
#nb = f_precisionp_metric(y, true_min, p)

def x_precisionp_metric(xmat, true_x_sol, p):
    
    """
     Return the number of iterations required before a point is sampled 
     with an objective function value within p*100% of the true solution
    Args:
        xmat (numpy.ndarray) : the data points so far, shape = (n, k)
        true_x_min (tuple) : true solution of the function, shape(1, k)
        p (0<float<1) : percentage of precision
    Returns:
        integer: nb of iterations required before a point is sampled 
        with an objective function value within p*100% of the true solution 
        (=0 if no such iteration)
    """
    n = xmat.shape[0]
    k = xmat.shape[1]
    
    spread_mat = spread_bylines(xmat, true_x_sol)
    nb_it_precisionp = 0
    
    for i in range(0,n):
        counter = 0
        for j in range(0, k):
            if spread_mat[i,j]>abs(p*true_x_sol[j]): break
            else: counter += 1
        if counter == k:
            nb_it_precisionp = i+1
            break
    
    return nb_it_precisionp

#test function x_precisionp_metric
#xmat=np.random.uniform(0,1,4).reshape(2,2)
#true_x_sol=np.array(xmat[1,]+0.0001).reshape(1,2)
#p=1/10
#x_precisionp_metric(xmat, true_x_sol, p)
    
def x_star_metric(xmat, true_x_sol):
    """
    The Euclidean distance from the nearest sample point to the global solution
    """
    nb_sample = xmat.shape[0]
    distances = np.zeros(nb_sample)
    for i in range(0,nb_sample):
        distances[i] = scipy.spatial.distance.euclidean(xmat[i,:], true_x_sol)
    index_min = np.argmin(distances)
    x_star = xmat[index_min,:]
    return scipy.spatial.distance.euclidean(x_star,true_x_sol)

#y=np.random.uniform(0,1,2).reshape(2,1)
#xmat=np.random.uniform(0,1,4).reshape(2,2)
#true_y_sol=min(y)/2
#true_x_sol=np.array(xmat[np.argmin(y),]+10).reshape(1,2)
#x_star_metric(y, xmat, true_y_sol, true_x_sol)
#scipy.spatial.distance.euclidean(xmat[np.argmin(y),:], xmat[np.argmin(y),]+10)
  
def rms_error(mat1, mat2):
    """
     Return the RMS error metric
    Args:
        mat1 (numpy.ndarray) : matrix, shape = (n, k)
        mat2 (numpy.ndarray) : matrix, shape = (n, k)
    Returns:
        float: RMS
    """
    N = np.size(mat1)
    return math.sqrt(np.sum((mat1-mat2)**2))/N

#test rms_erro
#vec1 = np.array([1,2])
#vec2 = np.array([1,3])
#rms_error(vec1, vec2)
    
def eval_fun_with_grid(functions2Beval, grid):
    """
     Return the RMS error metric
    Args:
        functions2Beval (tuple(function1, function2, ...)) : 
            tuple of l functions that MUST take 2 arguments
        grid (tuple(numpy.ndarray, numpy.ndarray)) : tuple of 2 matrix, 
            shape = (n, k) for each matrix
    Returns:
        numpy.ndarray(n,k,l): matrix forevaluations of each function
            on each point of the grid
    """
    
    nb_fun = len(functions2Beval)
    dim1, dim2 = np.shape(grid[0])
    results = np.zeros((dim1,dim2,nb_fun))
    for i in range(0, dim1):
        for j in range(0, dim2):
            for k in range(0, nb_fun):
                point_to_be_eval = np.array([grid[0][i,j],grid[1][i,j]]).reshape(2,1)
                results[i,j,k] = functions2Beval[k](point_to_be_eval)
    return results

#test eval_fun_with_grid
#grid = viz.mesh_grid([[0,5],[0,5]], [5,5])
#def sq(x,y): return x+y
#def sq1(x,y): return -x-y
#mat1 = eval_fun_with_grid([sq,sq1], grid)[:,:,0]
#mat2 = eval_fun_with_grid([sq,sq1], grid)[:,:,1]
#mat1-mat2
#(mat1-mat2)**2
#np.sum((mat1-mat2)**2)
#rms_error(mat1,mat2)
#np.size(mat1)

def rms_metric(prediction_function, true_function, bounds, gridsize):
    grid = viz.mesh_grid(bounds, gridsize)
    results_mat = eval_fun_with_grid([prediction_function,true_function], grid)
    error = rms_error(results_mat[:,:,0], results_mat[:,:,1])
    return error

#tet rms_metric
#bounds = [[0,5],[0,5]]
#gridsize = [5,5]
#def uno(x,y): return 1
#def dos(x,y): return 2
#rms_metric(uno, dos, bounds, gridsize)
#grid = viz.mesh_grid(bounds, gridsize)
#results_mat = eval_fun_with_grid([uno,dos], grid)
#error = rms_error(results_mat[:,0], results_mat[:,1])
    
def prediction_function_krigging(xnew, y, xmat, theta_vec, p_vec):
    R = exp_kernel.kernel_mat(xmat, theta_vec, p_vec)
    Rinv = cho_inv.cholesky_inv(R)
    beta_hat = pred.beta_est(y, Rinv)
    rx = exp_kernel.kernel_rx(xmat, xnew, theta_vec, p_vec)
    y_hat = pred.y_est(rx, y, Rinv, beta_hat)
    return y_hat

def all_metrics(true_function, bounds, gridsize,
                y, xmat, true_y_sol, true_x_sol, theta_vec, p_vec, p):
    
    def prediction_function_krigging_rms(xnew):
        return prediction_function_krigging(xnew, y, xmat, theta_vec, p_vec)
    
#    results = dict()
#    results["Best est."] = min(y)
#    results["f%"] = f_precisionp_metric(y, true_y_sol, p)
#    results["x%"] = x_precisionp_metric(xmat, true_x_sol, p)
#    results["x*"] = x_star_metric(xmat, true_x_sol)
#    results["RMS"] = rms_metric(prediction_function_krigging_rms, true_function, 
#                       bounds, gridsize)
    results = [float(min(y)), f_precisionp_metric(y, true_y_sol, p),
          x_precisionp_metric(xmat, true_x_sol, p),
          x_star_metric(xmat, true_x_sol), 
          rms_metric(prediction_function_krigging_rms,true_function, bounds, gridsize)]
    return results

#p=2/100
#def dist_double(x,y): return 2*scipy.spatial.distance.euclidean(x,y)
#bounds = [[0,5],[0,5]]
#gridsize = [5,5]
#y=np.random.uniform(0,1,3).reshape(3,1)
#xmat=np.random.uniform(0,1,6).reshape(3,2)
#true_y_sol=min(y)/2
#true_x_sol=np.array(xmat[2,]+0.0001).reshape(1,2)
#results = all_metrics(uno, dos, bounds, gridsize, 
#            y, xmat, true_y_sol, true_x_sol, p=0.5)


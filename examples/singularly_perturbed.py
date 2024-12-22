# Example file for the poisson problem
# Path: examples/poisson.py
# file contains the exact solution, rhs and boundary conditions for the poisson problem
import numpy as np
import tensorflow as tf

def left_boundary(x, y):
    """
    This function will return the boundary value for given component of a boundary
    """
    return 0.0

def right_boundary(x, y):
    """
    This function will return the boundary value for given component of a boundary
    """
    val = 0.0
    return val

def top_boundary(x, y):
    """
    This function will return the boundary value for given component of a boundary
    """
    val = 0.0
    return np.ones_like(x) * val

def bottom_boundary(x, y):
    """
    This function will return the boundary value for given component of a boundary
    """
    return 0.0

def rhs(x, y):
    """
    This function will return the value of the rhs at a given point
    """
    return 1.0 * np.ones_like(x)

def exact_solution(x, y):
    """
    This function will return the exact solution at a given point
    """
    eps = 1e-7
    # val = 16 * x * (1 - x) * y * (1 - y)
    val_x = x - (np.exp( -1.0 * ((1 - x)/eps)) - np.exp( -1.0 * (x/eps)) )/ (1 - np.exp(-1.0/eps))
    val_y = y - np.exp( -1.0 * ((1 - y)/eps)) - np.exp( -1.0 * (y/eps))/ (1 - np.exp(-1.0/eps))


    return val_x * val_y

def get_boundary_function_dict():
    """
    This function will return a dictionary of boundary functions
    """
    return {1000: bottom_boundary, 1001: right_boundary, 1002: top_boundary, 1003: left_boundary}

def get_bound_cond_dict():
    """
    This function will return a dictionary of boundary conditions
    """
    return {1000: "dirichlet", 1001: "dirichlet", 1002: "dirichlet", 1003: "dirichlet"}

def get_bilinear_params_dict():
    """
    This function will return a dictionary of bilinear parameters
    """
    eps = tf.constant(1.0, dtype=tf.float32)

    return {"eps": eps}
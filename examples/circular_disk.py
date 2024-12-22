# Example file for the poisson problem
# Path: examples/poisson.py
# file contains the exact solution, rhs and boundary conditions for the poisson problem
import numpy as np
import tensorflow as tf

def circular_boundary(x, y):
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
    eps = 1.0

    return np.ones_like(x) * 0.0

def get_boundary_function_dict():
    """
    This function will return a dictionary of boundary functions
    """
    return {1000: circular_boundary}

def get_bound_cond_dict():
    """
    This function will return a dictionary of boundary conditions
    """
    return {1000: "dirichlet"}

def get_bilinear_params_dict():
    """
    This function will return a dictionary of bilinear parameters
    """
    eps = tf.constant(1.0, dtype=tf.float32)

    return {"eps": eps}
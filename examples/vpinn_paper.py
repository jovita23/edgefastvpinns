# Example file for the poisson problem
# Path: examples/poisson.py
# file contains the exact solution, rhs and boundary conditions for the poisson problem
import numpy as np
import tensorflow as tf

def left_boundary(x, y):
    """
    This function will return the boundary value for given component of a boundary
    """
    val = np.tanh(20*x) * np.sin(np.pi * 2 * x * y)
    return val

def right_boundary(x, y):
    """
    This function will return the boundary value for given component of a boundary
    """
    val = np.tanh(20*x) * np.sin(np.pi * 2 * x * y)
    return val

def top_boundary(x, y):
    """
    This function will return the boundary value for given component of a boundary
    """
    val = np.tanh(20*x) * np.sin(np.pi * 2 * x * y)
    return val

def bottom_boundary(x, y):
    """
    This function will return the boundary value for given component of a boundary
    """
    val = np.tanh(20*x) * np.sin(np.pi * 2 * x * y)
    return val

def rhs(x, y):
    """
    This function will return the value of the rhs at a given point
    """
    # f_temp =  32 * (x  * (1 - x) + y * (1 - y))
    # f_temp = 1

    return -1.0 * (-4 * np.pi**2 * x**2 * np.sin(2 * np.pi * x * y) * np.tanh(20 * x) 
            - 4 * np.pi**2 * y**2 * np.sin(2 * np.pi * x * y) * np.tanh(20 * x) 
            + 4 * np.pi * y * (20 - 20 * np.tanh(20 * x)**2) * np.cos(2 * np.pi * x * y) 
            - 20 * (40 - 40 * np.tanh(20 * x)**2) * np.sin(2 * np.pi * x * y) * np.tanh(20 * x))

def exact_solution(x, y):
    """
    This function will return the exact solution at a given point
    """
    
    val = np.tanh(20*x) * np.sin(np.pi * 2 * x * y)

    return val

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
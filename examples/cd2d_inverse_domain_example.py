# Example file for the poisson problem
# Path: examples/poisson.py
# file contains the exact solution, rhs and boundary conditions for the poisson problem
# Actual Solution : sin(X) * cos(Y) * exp(-1.0 * eps * (X**2 + Y**2))
import numpy as np
import tensorflow as tf




def left_boundary(x, y):
    """
    This function will return the boundary value for given component of a boundary
    """
    val = np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)
    return val

def right_boundary(x, y):
    """
    This function will return the boundary value for given component of a boundary
    """
    val = np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)
    return val

def top_boundary(x, y):
    """
    This function will return the boundary value for given component of a boundary
    """
    val = np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)
    return val

def bottom_boundary(x, y):
    """
    This function will return the boundary value for given component of a boundary
    """
    val = np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)
    return val

def rhs(x, y):
    """
    This function will return the value of the rhs at a given point
    """

    # return -y * np.sin(x * y) * np.tanh(8 * x * y) + 8 * y * np.cos(x * y) / np.cosh(8 * x * y)**2 + 10 * (x**2 + y**2) * \
    #       (16 * np.sin(x * y) / np.cosh(8 * x * y)**2 + np.cos(x * y) * np.tanh(8 * x * y) + 128 * np.cos(x * y) * np.tanh(8 * x * y) / np.cosh(8 * x * y)**2) * np.sin(x) * np.cos(y)

    
    return 128*np.pi**2*x*y*(1 - x)*(1 - y)*np.sin(2*np.pi*x)*np.sin(2*np.pi*y) + 2*np.pi*np.sin(2*np.pi*y)*np.cos(2*np.pi*x)


def exact_solution(x, y):
    """
    This function will return the exact solution at a given point
    """
    
    val = np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)

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
    
    eps = 0.1 # will not be used in the loss function, as it will be replaced by the predicted value of NN
    b1 = 1
    b2 = 0

    return {"eps": eps, "b_x": b1, "b_y": b2}



def get_inverse_params_actual_dict(x, y):
    """
    This function will return a dictionary of inverse parameters
    """
    # Initial Guess
    eps = 16*x*y*(1-x)*(1-y)
    return {"eps": eps}
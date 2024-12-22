# Example file for the poisson problem
# Path: examples/poisson.py
# file contains the exact solution, rhs and boundary conditions for the poisson problem
import numpy as np
import tensorflow as tf
import sympy as sp


EPS = 0.1

def left_boundary(x, y):
    """
    This function will return the boundary value for given component of a boundary
    """
    val = np.sin(x) * np.tanh(x) * np.exp(-1.0*EPS *(x**2)) * 10
    return val



def rhs(x, y):
    """
    This function will return the value of the rhs at a given point
    """

    X =x
    Y =y
    eps = EPS

    return -EPS * (
        40.0 * X * eps * (np.tanh(X)**2 - 1) * np.sin(X) 
        - 40.0 * X * eps * np.cos(X) * np.tanh(X) 
        + 10 * eps * (4.0 * X**2 * eps - 2.0) * np.sin(X) * np.tanh(X) 
        + 20 * (np.tanh(X)**2 - 1) * np.sin(X) * np.tanh(X) 
        - 20 * (np.tanh(X)**2 - 1) * np.cos(X) 
        - 10 * np.sin(X) * np.tanh(X)
    ) * np.exp(-1.0 * X**2 * eps)

def exact_solution(x, y):
    """
    This function will return the exact solution at a given point
    """
    
    val = np.sin(x) * np.tanh(x) * np.exp(-1.0*EPS *(x**2)) * 10

    return val

def get_boundary_function_dict():
    """
    This function will return a dictionary of boundary functions
    """
    return {1000: left_boundary}

def get_bound_cond_dict():
    """
    This function will return a dictionary of boundary conditions
    """
    return {1000: "dirichlet"}

def get_bilinear_params_dict():
    """
    This function will return a dictionary of bilinear parameters
    """
    eps = 1.0

    return {"eps": eps}
# Example file for the poisson problem
# Path: examples/poisson.py
# file contains the exact solution, rhs and boundary conditions for the poisson problem
# Actual Solution : sin(X) * cos(Y) * exp(-1.0 * eps * (X**2 + Y**2))
import numpy as np
import tensorflow as tf


EPS = 1.0

def left_boundary(x, y):
    """
    This function will return the boundary value for given component of a boundary
    """
    val = np.sin(x) * np.cos(y) * np.exp(-1.0 * EPS * (x**2 + y**2)) 
    return val

def right_boundary(x, y):
    """
    This function will return the boundary value for given component of a boundary
    """
    val = np.sin(x) * np.cos(y) * np.exp(-1.0 * EPS * (x**2 + y**2)) 
    return val

def top_boundary(x, y):
    """
    This function will return the boundary value for given component of a boundary
    """
    val = np.sin(x) * np.cos(y) * np.exp(-1.0 * EPS * (x**2 + y**2)) 
    return val

def bottom_boundary(x, y):
    """
    This function will return the boundary value for given component of a boundary
    """
    val = np.sin(x) * np.cos(y) * np.exp(-1.0 * EPS * (x**2 + y**2)) 
    return val

def rhs(x, y):
    """
    This function will return the value of the rhs at a given point
    """

    X =x
    Y =y

    return -(-4.0 * X * EPS * np.cos(X) + EPS * (4.0 * X**2 * EPS - 2.0) * np.sin(X) - np.sin(X)) * np.exp(-1.0 * EPS * (X**2 + Y**2)) * np.cos(Y) - \
           (4.0 * Y * EPS * np.sin(Y) + EPS * (4.0 * Y**2 * EPS - 2.0) * np.cos(Y) - np.cos(Y)) * np.exp(-1.0 * EPS * (X**2 + Y**2)) * np.sin(X)


def exact_solution(x, y):
    """
    This function will return the exact solution at a given point
    """
    
    val = np.sin(x) * np.cos(y) * np.exp(-1.0 * EPS * (x**2 + y**2)) 

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
    # Initial Guess
    eps = EPS

    return {"eps": eps}


def get_inverse_params_dict():
    """
    This function will return a dictionary of inverse parameters
    """
    # Initial Guess
    eps = 0.1

    return {"eps": eps}


def get_inverse_params_actual_dict():
    """
    This function will return a dictionary of inverse parameters
    """
    # Initial Guess
    eps = EPS

    return {"eps": eps}
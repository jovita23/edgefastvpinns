# Example file for the poisson problem
# Path: examples/poisson.py
# file contains the exact solution, rhs and boundary conditions for the poisson problem
# History : UNDER DEVELOPMENT - DO NOT USE

import numpy as np
import tensorflow as tf


print("Example file for the poisson problem - under development - DO NOT USE")
exit()

from sympy import symbols, Piecewise, sin, pi, cos, integrate

#import atan2
from math import atan2

# Define the symbols
r, theta, phi = symbols('r theta phi')

# Define the function h(phi)
def h(phi):
    val = np.where(phi <= np.pi, np.sin(phi), 0)
    return val


def circular_boundary(x, y):
    """
    This function will return the boundary value for given component of a boundary
    """
    r_val = (x**2 + y**2)**0.5
    theta_val = np.arctan2(y, x)
    
    # Apply the boundary condition function h(theta)
    return h(theta_val)

def rhs(x, y):
    """
    This function will return the value of the rhs at a given point
    """
    return 0.0 * np.ones_like(x)

def exact_solution(x, y):
    """
    This function will return the exact solution at a given point
    """


    # Calculate the condition
    condition = x**2 + y**2 < 0.99

    r_val = (x**2 + y**2)**0.5
    theta_val = np.arctan2(y, x)

    # Calculate the values
    val1 = np.where(condition, y/2, h(theta_val))
    val2 = np.where(condition, x * (1 - x**2 - y**2) / (4 * np.pi * (x**2 + y**2)), 0)
    num = (1 + x**2) + y**2
    denom = ( 1 - x**2 ) + y**2
    val3 = np.where(condition, np.log(num/denom), 0)
    val4 = np.where(condition, y * ( 1 + x**2 + y**2) / (2 * np.pi * (x**2 + y**2)), 0)
    val5 = np.where(condition, np.arctan((2*y)/(1 - x**2 - y**2)), 0)

    # Calculate the final value
    val = val1 + val2*val3 + val4*val5

    

    return val


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
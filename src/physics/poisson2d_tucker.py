# This file contains the loss function for the poisson problem in 2D domain
# this loss function will be passed as a class to the tensorflow custom model
# Author : Thivin Anandh
# Date : 22/Sep/2023
# History : 22/Sep/2023 - Initial implementation with basic loss function
import tensorflow as tf

import tensorly as tl
from tensorly.decomposition import tucker



import numpy as np

@tf.function
def mode_dot(tensor, matrix, mode):
    # Move the mode to the first dimension
    tensor = tf.experimental.numpy.moveaxis(tensor, mode, 0)
    # Perform the dot product
    result = tf.tensordot(matrix, tensor, axes=(1, 0))
    # Move the first dimension back to its original place
    result = tf.experimental.numpy.moveaxis(result, 0, mode)
    return result


# PDE loss function for the poisson problem
#@tf.function
def pde_loss_poisson(test_shape_val_mat, test_grad_x_mat, test_grad_y_mat, pred_nn, pred_grad_x_nn, pred_grad_y_nn, forcing_function, bilinear_params):
    """
    This method returns the loss for the Poisson Problem of the PDE
    """
    # Compute PDE loss
    
   
    core = tf.convert_to_tensor(test_grad_x_mat[0])
    factors = list(test_grad_x_mat[1])    

    factors_0 = tf.convert_to_tensor(factors[0])
    factors_1 = tf.convert_to_tensor(factors[1])
    factors_2 = tf.convert_to_tensor(factors[2])

   

    test_x = tl.tucker_to_tensor((core,factors))

    
    
    r = tf.transpose(factors_2)@tf.transpose(pred_grad_x_nn)
    #r1 = np.tensordot(core,r,axes=([2],[0]))
    r1 = tf.tensordot(core, r, axes=([2],[0]))
    # r2 = tl.tenalg.mode_dot(r1,factors_1,mode = 1)
    r2 = mode_dot(r1,factors_1, mode=1)
    r2_reshaped = tf.transpose(r2, perm=[1, 2, 0])
    pde_diffusion_x = tf.einsum('ijk,kj->ij', r2_reshaped, tf.transpose(factors_0))

    

   


    core = tf.convert_to_tensor(test_grad_y_mat[0])
    factors = test_grad_y_mat[1]

    factors_0 = tf.convert_to_tensor(factors[0])
    factors_1 = tf.convert_to_tensor(factors[1])
    factors_2 = tf.convert_to_tensor(factors[2])

    test_y = tl.tucker_to_tensor((core,factors))

    r = tf.transpose(factors_2)@tf.transpose(pred_grad_y_nn)
    #r1 = np.tensordot(core,r,axes=([2],[0]))
    r1 = tf.tensordot(core, r, axes=([2],[0]))
    # r2 = tl.tenalg.mode_dot(r1,factors_1,mode = 1)
    r2 = mode_dot(r1,factors_1, mode=1)
    r2_reshaped = tf.transpose(r2, perm=[1, 2, 0])
    pde_diffusion_y = tf.einsum('ijk,kj->ij', r2_reshaped, tf.transpose(factors_0))
  
    pde_diffusion_x_actual = tf.transpose(tf.linalg.matvec(test_x, pred_grad_x_nn))
   
    pde_diffusion_y_actual = tf.transpose(tf.linalg.matvec(test_y, pred_grad_y_nn))


    pde_diffusion = bilinear_params["eps"] * (pde_diffusion_x + pde_diffusion_y)
    pde_diffusion_actual = bilinear_params["eps"] * (pde_diffusion_x_actual + pde_diffusion_y_actual)

    diff = pde_diffusion - pde_diffusion_actual

    norm = tf.norm(diff,ord=1)
    print(norm)


    #residual_matrix = pde_diffusion - forcing_function
    residual_matrix_actual = pde_diffusion_actual - forcing_function


    residual_cells = tf.reduce_mean(tf.square(residual_matrix_actual), axis=0)
    
    

    return residual_cells

#This file contains the loss function for the CD2D problem 
#The loss tensor is decomposed using the cp decomposition 

import tensorflow as tf



@tf.function
def pde_loss_cd2d(test_shape_val_mat, test_grad_x_mat, test_grad_y_mat, pred_nn, pred_grad_x_nn, pred_grad_y_nn, forcing_function, bilinear_params):

   
    factors_diff_x = list(test_grad_x_mat[1])    
    factors_diff_y = list(test_grad_y_mat[1])   


    result_diff_x = tf.einsum('kr, ik -> ir', factors_diff_x[2],pred_grad_x_nn)
    result_diff_x = tf.einsum('ir, jr -> ijr', result_diff_x,factors_diff_x[1])  #r should be less than k
    result_diff_x = tf.einsum('ijr, ir -> ij', result_diff_x, factors_diff_x[0])

    pde_diffusion_x = tf.transpose(result_diff_x)

   
    result_diff_y = tf.einsum('kr, ik -> ir', factors_diff_y[2],pred_grad_y_nn)
    result_diff_y = tf.einsum('ir, jr -> ijr', result_diff_y,factors_diff_y[1])
    result_diff_y = tf.einsum('ijr, ir -> ij', result_diff_y, factors_diff_y[0])

    pde_diffusion_y = tf.transpose(result_diff_y)

    
  
    result_conv_x = tf.einsum('kr, ik -> ir',factors_diff_x[2], pred_nn)
    result_conv_x = tf.einsum('ir, jr -> ijr', result_conv_x, factors_diff_x[1])
    result_conv_x = tf.einsum('ijr, ir -> ij', result_conv_x, factors_diff_x[0])

    conv_x = tf.transpose(result_conv_x)



    result_conv_y = tf.einsum('kr, ik -> ir',factors_diff_y[2], pred_nn)
    result_conv_y = tf.einsum('ir, jr -> ijr', result_conv_y, factors_diff_y[1])
    result_conv_y = tf.einsum('ijr, ir -> ij', result_conv_y, factors_diff_y[0])

    conv_y = tf.transpose(result_conv_y)

    
    
    conv = bilinear_params["b_x"] * conv_x + bilinear_params["b_y"] * conv_y

    pde_diffusion = bilinear_params["eps"] * (pde_diffusion_x + pde_diffusion_y)

    residual_matrix = (pde_diffusion - conv) - forcing_function


    residual_cells = tf.reduce_mean(tf.square(residual_matrix), axis=0)
    

    return residual_cells
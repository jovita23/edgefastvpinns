#This file contains the loss function for the CD2D problem 
#The loss tensor is decomposed using the tucker decomposition 

import tensorflow as tf

@tf.function
def pde_loss_cd2d(test_shape_val_mat, test_grad_x_mat, test_grad_y_mat, pred_nn, pred_grad_x_nn, pred_grad_y_nn, forcing_function, bilinear_params,  inverse_params_list):

    diffusion_coeff_NN = inverse_params_list[0]



    core = tf.convert_to_tensor(test_grad_x_mat[0])
    factors = list(test_grad_x_mat[1])    


    result = tf.einsum('mnp, kp -> mnk', core, factors[2])
    result = tf.einsum('mnk, ik -> mni', result, pred_grad_x_nn*diffusion_coeff_NN)

   
    result = tf.einsum('mni, jn -> imj', result, factors[1])  
    result = tf.einsum('imj, im -> ij', result, factors[0])

    pde_diffusion_x = tf.transpose(result)


    core = tf.convert_to_tensor(test_grad_y_mat[0])
    factors = list(test_grad_y_mat[1])    


    result = tf.einsum('mnp, kp -> mnk', core, factors[2])
    result = tf.einsum('mnk, ik -> mni', result, pred_grad_y_nn*diffusion_coeff_NN)

   
    result = tf.einsum('mni, jn -> imj', result, factors[1])
    result = tf.einsum('imj, im -> ij', result, factors[0])

    pde_diffusion_y = tf.transpose(result)

    
    core = tf.convert_to_tensor(test_grad_x_mat[0])
    factors = list(test_grad_x_mat[1])   

    result = tf.einsum('mnp, kp -> mnk', core, factors[2])
    result = tf.einsum('mnk, ik -> mni', result, pred_nn)

   
    result = tf.einsum('mni, jn -> imj', result, factors[1])
    result = tf.einsum('imj, im -> ij', result, factors[0])

    conv_x = tf.transpose(result)

    core = tf.convert_to_tensor(test_grad_y_mat[0])
    factors = list(test_grad_y_mat[1])   


    result = tf.einsum('mnp, kp -> mnk', core, factors[2])
    result = tf.einsum('mnk, ik -> mni', result, pred_nn)

   
    result = tf.einsum('mni, jn -> imj', result, factors[1])
    result = tf.einsum('imj, im -> ij', result, factors[0])

    conv_y = tf.transpose(result)
    conv = bilinear_params["b_x"] * conv_x + bilinear_params["b_y"] * conv_y

    pde_diffusion = pde_diffusion_x + pde_diffusion_y

    residual_matrix = pde_diffusion - forcing_function


    residual_cells = tf.reduce_mean(tf.square(residual_matrix), axis=0)
    

    return residual_cells
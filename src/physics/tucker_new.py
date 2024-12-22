# Author : Thivin Anandh, Jovita Biju
import tensorflow as tf

@tf.function
def pde_loss_poisson(test_shape_val_mat, test_grad_x_mat, test_grad_y_mat, pred_nn, pred_grad_x_nn, pred_grad_y_nn, forcing_function, bilinear_params):

    
    core_diff_x = test_grad_x_mat[0]
    factors_diff_x = list(test_grad_x_mat[1])    
    
    


    result_diff_x = tf.einsum('mnp, kp -> mnk', core_diff_x, factors_diff_x[2])
    result_diff_x = tf.einsum('mnk, ik -> mni', result_diff_x, pred_grad_x_nn)

   
    result_diff_x = tf.einsum('mni, jn -> imj', result_diff_x, factors_diff_x[1])
    result_diff_x = tf.einsum('imj, im -> ij', result_diff_x, factors_diff_x[0])

    pde_diffusion_x = tf.transpose(result_diff_x)


    core_diff_y = test_grad_y_mat[0]
    factors_diff_y = list(test_grad_y_mat[1])    


    result_diff_y = tf.einsum('mnp, kp -> mnk', core_diff_y, factors_diff_y[2])
    result_diff_y = tf.einsum('mnk, ik -> mni', result_diff_y, pred_grad_y_nn)

   
    result_diff_y = tf.einsum('mni, jn -> imj', result_diff_y, factors_diff_y[1])
    result_diff_y = tf.einsum('imj, im -> ij', result_diff_y, factors_diff_y[0])

    pde_diffusion_y = tf.transpose(result_diff_y)

    pde_diffusion = bilinear_params["eps"] * (pde_diffusion_x + pde_diffusion_y)

    residual_matrix = pde_diffusion - forcing_function


    residual_cells = tf.reduce_mean(tf.square(residual_matrix), axis=0)
    

   
    

    return residual_cells
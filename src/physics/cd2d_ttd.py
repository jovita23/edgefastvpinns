#This file contains the loss function for the CD2D problem 
#The loss tensor is decomposed using the tensor train decomposition decomposition 

import tensorflow as tf
import t3f

@tf.function
def pde_loss_cd2d(test_shape_val_mat, test_grad_x_mat, test_grad_y_mat, pred_nn, pred_grad_x_nn, pred_grad_y_nn, forcing_function, bilinear_params):
    
    ttd_cores_x = test_grad_x_mat[1]
    ranks = t3f.shapes.lazy_tt_ranks(ttd_cores_x)

   
   
    tt_cores_2_x= tf.reshape(ttd_cores_x.tt_cores[2],(ranks[2],-1))
    tt_cores_1_x= ttd_cores_x.tt_cores[1]
    tt_cores_0_x= tf.reshape(ttd_cores_x.tt_cores[0],(-1,ranks[1]))


    result_diff_x_1 = tf.einsum('ij, jkr -> ikr', tt_cores_0_x,tt_cores_1_x)
    result_diff_x_2 = tf.einsum('rk, ik -> ir',tt_cores_2_x,pred_grad_x_nn)
    result_diff_x   = tf.einsum('ir, ijr -> ij',result_diff_x_2, result_diff_x_1)

    pde_diffusion_x = tf.transpose(result_diff_x)


    ttd_cores_y = test_grad_y_mat[1]

    tt_cores_2_y= tf.reshape(ttd_cores_y.tt_cores[2],(ranks[2],-1))
    tt_cores_1_y=  ttd_cores_y.tt_cores[1]
    tt_cores_0_y= tf.reshape( ttd_cores_y.tt_cores[0],(-1,ranks[1]))

    result_diff_y_1 = tf.einsum('ij, jkr -> ikr', tt_cores_0_y,tt_cores_1_y)
    result_diff_y_2 = tf.einsum('rk, ik -> ir',tt_cores_2_y,pred_grad_y_nn)
    result_diff_y   = tf.einsum('ir, ijr -> ij',result_diff_y_2, result_diff_y_1)
    pde_diffusion_y = tf.transpose(result_diff_y)

    
  
    result_conv_x_1 = tf.einsum('ij, jkr -> ikr', tt_cores_0_x,tt_cores_1_x)
    result_conv_x_2 = tf.einsum('rk, ik -> ir',tt_cores_2_x,pred_nn)
    result_conv_x   = tf.einsum('ir, ijr -> ij',result_diff_x_2, result_diff_x_1)
    conv_x = tf.transpose(result_conv_x)



    result_conv_y_1 = tf.einsum('ij, jkr -> ikr', tt_cores_0_y,tt_cores_1_y)
    result_conv_y_2 = tf.einsum('rk, ik -> ir',tt_cores_2_y,pred_nn)
    result_conv_y   = tf.einsum('ir, ijr -> ij',result_diff_y_2, result_diff_y_1)
    conv_y = tf.transpose(result_conv_y)
    
    
    conv = bilinear_params["b_x"] * conv_x + bilinear_params["b_y"] * conv_y

    pde_diffusion = bilinear_params["eps"] * (pde_diffusion_x + pde_diffusion_y)

    residual_matrix = (pde_diffusion - conv) - forcing_function


    residual_cells = tf.reduce_mean(tf.square(residual_matrix), axis=0)
    print(residual_cells)

    return residual_cells
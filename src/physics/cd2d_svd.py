# This file contains the loss function for the cd2d problem in 2D domain
# this loss function will be passed as a class to the tensorflow custom model
# Author : Thivin Anandh
# Date : 22/Sep/2023
# History : 22/Sep/2023 - Initial implementation with basic loss function
import tensorflow as tf

# PDE loss function for the cd2d problem
@tf.function
def pde_loss_cd2d_svd(test_shape_val_mat, test_grad_x_mat, test_grad_y_mat, pred_nn, pred_grad_x_nn, pred_grad_y_nn, forcing_function, bilinear_params):
    """
    This method returns the loss for the CD2D Problem of the PDE
    """
    test_U = test_shape_val_mat[0]
    test_VT = test_shape_val_mat[1]

    test_grad_x_U = test_grad_x_mat[0]
    test_grad_x_VT = test_grad_x_mat[1]

    test_grad_y_U = test_grad_y_mat[0]
    test_grad_y_VT = test_grad_y_mat[1]

    # Compute PDE loss
    pde_diffusion_x = tf.linalg.matvec(test_grad_x_VT, pred_grad_x_nn)
    pde_diffusion_x = tf.transpose(tf.linalg.matvec(test_grad_x_U, pde_diffusion_x))


    pde_diffusion_y = tf.linalg.matvec(test_grad_y_VT, pred_grad_y_nn)
    pde_diffusion_y = tf.transpose(tf.linalg.matvec(test_grad_y_U, pde_diffusion_y))

    conv_x = tf.linalg.matvec(test_VT, pred_grad_x_nn)
    conv_x = tf.transpose(tf.linalg.matvec(test_U, conv_x))


    conv_y = tf.linalg.matvec(test_VT, pred_grad_y_nn)
    conv_y = tf.transpose(tf.linalg.matvec(test_U, conv_y))


    # # b(x) * ∫du/dx. v dΩ + b(y) * ∫du/dy. v dΩ
    conv = bilinear_params["b_x"] * conv_x + bilinear_params["b_y"] * conv_y

    pde_diffusion = bilinear_params["eps"] * (pde_diffusion_x + pde_diffusion_y)




    residual_matrix = (pde_diffusion + conv ) - forcing_function

    # Perform Reduce mean along the axis 0
    residual_cells = tf.reduce_mean(tf.square(residual_matrix), axis=0)


    return residual_cells
 
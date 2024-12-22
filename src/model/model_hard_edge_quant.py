import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import t3f

# Custom Loss Functions
def custom_loss1(y_true1, y_pred1):
    return tf.reduce_mean(tf.square(y_pred1 - y_true1))

def custom_loss2(y_true2, y_pred2):
    return tf.reduce_mean(tf.square(y_pred2 - y_true2))

# class CustomKerasDense(t3f.nn.KerasDense):
#     def __init__(self, input_dims, output_dims, tt_rank, activation='relu',  bias_initializer='zeros'):
#         super(CustomKerasDense, self).__init__(input_dims, output_dims, tt_rank, activation, bias_initializer)
#         self.input_dims = input_dims  
#         self.output_dims = output_dims
#         self.tt_rank = tt_rank
#         self.activation = activation
#         self.bias_initializer = bias_initializer

#     def get_config(self):
#         config = super().get_config()
#         config.update({
#             'input_dims': self.input_dims,
#             'output_dims': self.output_dims,
#             'tt_rank': self.tt_rank,
#             'activation': self.activation,
#             'bias_initializer': self.bias_initializer
#         })
#         return config


# Define the functional model
def create_functional_model(layer_dims, input_shape, activation='tanh', tensor_dtype=tf.float32):
    #inputs = layers.Input(shape=input_shape, dtype=tensor_dtype)
    #x = layers.Dense(30, activation=activation)(inputs)

    input_tensor = layers.Input(shape=(layer_dims[0],), dtype=tensor_dtype)

    # Build dense layers based on the input list
    x = input_tensor

   
    for dim in range(len(layer_dims) - 2):
        x = layers.Dense(layer_dims[dim+1], activation=activation, kernel_initializer='glorot_uniform', dtype=tensor_dtype, bias_initializer='zeros')(x)

    # Output layer with no activation
    output_tensor = layers.Dense(layer_dims[-1], activation=None, kernel_initializer='glorot_uniform', dtype=tensor_dtype, bias_initializer='zeros')(x)

   
    
    model = models.Model(inputs=input_tensor, outputs=output_tensor)
    return model

# Custom training loop
class CustomTrainingLoop(tf.keras.Model):
    def __init__(self, model, loss_function, input_tensors_list, orig_factor_matrices, force_function_list, params_dict, learning_rate_dict, tensor_dtype):
        super(CustomTrainingLoop, self).__init__()
        self.model = model
        self.loss_function = loss_function
        self.input_tensors_list = input_tensors_list
        self.input_tensor = input_tensors_list[0]
        self.dirichlet_input = input_tensors_list[1]
        self.dirichlet_actual = input_tensors_list[2]
        self.orig_factor_matrices = orig_factor_matrices
        self.shape_function_mat_list = orig_factor_matrices[0]
        self.shape_function_grad_x_factor_mat_list = orig_factor_matrices[1]
        self.shape_function_grad_y_factor_mat_list = orig_factor_matrices[2]
        self.force_function_list = force_function_list
        self.params_dict = params_dict
        self.n_cells = params_dict['n_cells']
        self.tensor_dtype = tensor_dtype
        
        initial_learning_rate = learning_rate_dict['initial_learning_rate']
        use_lr_scheduler = learning_rate_dict['use_lr_scheduler']
        decay_steps = learning_rate_dict['decay_steps']
        decay_rate = learning_rate_dict['decay_rate']
        staircase = learning_rate_dict['staircase']
        
        if use_lr_scheduler:
            learning_rate_fn = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate, decay_steps, decay_rate, staircase=True
            )
        else:
            learning_rate_fn = initial_learning_rate
            
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)

    def train_step(self, beta=10, bilinear_params_dict=None):
        with tf.GradientTape(persistent=True) as tape:
            predicted_values_dirichlet = self.model(self.dirichlet_input)
            total_pde_loss = 0.0

            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch(self.input_tensor)
                predicted_values = self.model(self.input_tensor)
            
            gradients = tape1.gradient(predicted_values, self.input_tensor)
            pred_grad_x = tf.reshape(gradients[:, 0], [self.n_cells, self.shape_function_grad_x_factor_mat_list.shape[-1]])
            pred_grad_y = tf.reshape(gradients[:, 1], [self.n_cells, self.shape_function_grad_y_factor_mat_list.shape[-1]])
            pred_val = tf.reshape(predicted_values, [self.n_cells, self.shape_function_mat_list.shape[-1]])
            
            cells_residual = self.loss_function(
                test_shape_val_mat=self.shape_function_mat_list, 
                test_grad_x_mat=self.shape_function_grad_x_factor_mat_list,
                test_grad_y_mat=self.shape_function_grad_y_factor_mat_list,
                pred_nn=pred_val,
                pred_grad_x_nn=pred_grad_x,
                pred_grad_y_nn=pred_grad_y,
                forcing_function=self.force_function_list,
                bilinear_params=bilinear_params_dict
            )
            
            residual = tf.reduce_sum(cells_residual)
            total_pde_loss += residual
            boundary_loss = tf.reduce_mean(tf.square(predicted_values_dirichlet - self.dirichlet_actual), axis=0)
            total_loss = total_pde_loss + beta * boundary_loss
        
        trainable_vars = self.model.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        return {"loss_pde": total_pde_loss, "loss_dirichlet": boundary_loss, "loss": total_loss}



# Main File for running the Python code
# For inverse problems
# Author: Thivin Anandh D
# Date:  19/Dec/2023


# import Libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm

import yaml
import sys
import os
import time

# set seeds
np.random.seed(42)
tf.random.set_seed(42)
tf.keras.utils.set_random_seed(42)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # or any {'0', '1', '2'}


# import the FE classes for 2D
from src.FE_2D.basis_function_2d import *
from src.FE_2D.fespace2d import *

# Import the Geometry class
from src.Geometry.geometry_2d import *

# import the model class
from src.model.model_inverse import *

# import the example file
from examples.inverse_tanh import *

#import physics for custom loss function
from src.physics.poisson2d_inverse import *

# import the data handler class
from src.data.datahandler2d import *

# import the plot utils
# from src.utils.plot_utils import *
from src.utils.edge_utils import *
from src.utils.compute_utils import *
from src.utils.print_utils import *


if __name__ == "__main__":
    # check input arguments
    if len(sys.argv) != 2:
        print("Usage: python main.py <config_file>")
        sys.exit(1)

    # Read the YAML file
    with open(sys.argv[1], 'r') as f:
        config = yaml.safe_load(f)
    

    # Extract the values from the YAML file
    i_output_path = config['experimentation']['output_path']

    i_mesh_generation_method = config['geometry']['mesh_generation_method']
    i_mesh_type = config['geometry']['mesh_type']
    i_x_min = config['geometry']['internal_mesh_params']['x_min']
    i_x_max = config['geometry']['internal_mesh_params']['x_max']
    i_y_min = config['geometry']['internal_mesh_params']['y_min']
    i_y_max = config['geometry']['internal_mesh_params']['y_max']
    i_n_cells_x = config['geometry']['internal_mesh_params']['n_cells_x']
    i_n_cells_y = config['geometry']['internal_mesh_params']['n_cells_y']
    i_n_boundary_points = config['geometry']['internal_mesh_params']['n_boundary_points']
    i_n_test_points_x = config['geometry']['internal_mesh_params']['n_test_points_x']
    i_n_test_points_y = config['geometry']['internal_mesh_params']['n_test_points_y']

    i_mesh_file_name = config['geometry']['external_mesh_params']['mesh_file_name']
    i_boundary_refinement_level = config['geometry']['external_mesh_params']['boundary_refinement_level']
    i_boundary_sampling_method = config['geometry']['external_mesh_params']['boundary_sampling_method']

    i_fe_order = config['fe']['fe_order']
    i_fe_type = config['fe']['fe_type']
    i_quad_order = config['fe']['quad_order']
    i_quad_type = config['fe']['quad_type']

    i_model_architecture = config['model']['model_architecture']
    i_activation = config['model']['activation']
    i_use_attention = config['model']['use_attention']
    i_epochs = config['model']['epochs']
    i_dtype = config['model']['dtype']
    if i_dtype == "float64":
        i_dtype = tf.float64
    elif i_dtype == "float32":
        i_dtype = tf.float32
    else:
        print("[ERROR] The given dtype is not a valid tensorflow dtype")
        raise ValueError("The given dtype is not a valid tensorflow dtype")
    
    i_set_memory_growth = config['model']['set_memory_growth']
    i_learning_rate_dict = config['model']['learning_rate']

    i_num_sensor_points = config['inverse']['num_sensor_points']

    i_beta = config['pde']['beta']

    i_update_progress_bar = config['logging']['update_progress_bar']
    i_update_console_output = config['logging']['update_console_output']
    i_update_solution_images = config['logging']['update_solution_images']
    i_last_test_error_epochs = config['logging']['last_test_error_epochs']


    i_use_wandb = config['wandb']['use_wandb']
    i_wandb_project_name = config['wandb']['project_name']
    i_wandb_run_prefix = config['wandb']['wandb_run_prefix']
    i_wandb_entity = config['wandb']['entity']

    # ---------------------------------------------------------------#
    
    # For expansion of GPU Memory
    if i_set_memory_growth:
        print("[INFO] Setting memory growth for GPU")
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)
 

    # use pathlib to create the folder,if it does not exist
    folder = Path(i_output_path)
    # create the folder if it does not exist
    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)

    # Initiate a Geometry_2D object
    domain = Geometry_2D(i_mesh_type, i_mesh_generation_method, i_n_test_points_x, i_n_test_points_y, i_output_path)

    # Read mesh from a .mesh file
    if i_mesh_generation_method == "external":
        cells, boundary_points = domain.read_mesh(mesh_file = i_mesh_file_name, \
                                                  boundary_point_refinement_level=i_boundary_refinement_level, \
                                                  bd_sampling_method=i_boundary_sampling_method, \
                                                  refinement_level=0)

    
    elif i_mesh_generation_method == "internal":
        cells, boundary_points = domain.generate_quad_mesh_internal(x_limits = [i_x_min, i_x_max], \
                                                                    y_limits = [i_y_min, i_y_max], \
                                                                    n_cells_x =  i_n_cells_x, \
                                                                    n_cells_y = i_n_cells_y, \
                                                                    num_boundary_points=i_n_boundary_points)

    # get the boundary function dictionary from example file
    bound_function_dict, bound_condition_dict = get_boundary_function_dict(), get_bound_cond_dict()
    
    # get fespace2d
    fespace = Fespace2D(mesh = domain.mesh, cells=cells, boundary_points=boundary_points, 
                        cell_type=domain.mesh_type, fe_order=i_fe_order, fe_type =i_fe_type ,quad_order=i_quad_order, quad_type = i_quad_type, \
                        fe_transformation_type="bilinear", bound_function_dict = bound_function_dict, \
                        bound_condition_dict = bound_condition_dict, \
                        forcing_function=rhs, output_path=i_output_path)

    # Instantiate the DataHandler2D class
    datahandler = DataHandler2D(fespace, domain, dtype=i_dtype)

    params_dict = {}
    params_dict['n_cells'] = fespace.n_cells

    # get the input data for the PDE
    train_dirichlet_input, train_dirichlet_output = datahandler.get_dirichlet_input()

    # get bilinear parameters
    # this function will obtain the values of the bilinear parameters from the model
    # and convert them into tensors of desired dtype
    bilinear_params_dict = datahandler.get_bilinear_params_dict_as_tensors(get_bilinear_params_dict)


    # Obtain sensor data
    points, sensor_values = datahandler.get_sensor_data(exact_solution, num_sensor_points=i_num_sensor_points, mesh_type=i_mesh_generation_method)
    
    # Obtain the inverse parameters
    inverse_params_dict = datahandler.get_inverse_params(get_inverse_params_dict)

    # obtain the target inverse parameters
    target_inverse_params_dict = get_inverse_params_actual_dict()

    # get actual Epsilon
    actual_epsilon = target_inverse_params_dict["eps"]


    model = DenseModel_Inverse(layer_dims = i_model_architecture, learning_rate_dict = i_learning_rate_dict, \
                       params_dict = params_dict, \
                       loss_function = pde_loss_poisson_inverse, input_tensors_list = [datahandler.x_pde_list, train_dirichlet_input, train_dirichlet_output], \
                        orig_factor_matrices = [datahandler.shape_val_mat_list , datahandler.grad_x_mat_list, datahandler.grad_y_mat_list], \
                        force_function_list=datahandler.forcing_function_list, \
                        sensor_list = [points, sensor_values], \
                        inverse_params_dict = inverse_params_dict, \
                        tensor_dtype = i_dtype,
                        use_attention=i_use_attention, \
                        activation=i_activation, \
                        hessian=False)
    

    # ---------------------------------------------------------------#
    # --------------    Get Testing points   ----------------------- #
    # ---------------------------------------------------------------#
    
    # test_points = np.c_[xx.ravel(), yy.ravel()]
    #code obtains the test points based on internal or external mesh
    test_points = domain.get_test_points()
    print(f"[bold]Number of Test Points = [/bold] {test_points.shape[0]}")
    y_exact = exact_solution(test_points[:,0], test_points[:,1])
    

    # save points for plotting
    if i_mesh_generation_method == "internal":
        X = test_points[:,0].reshape(i_n_test_points_x, i_n_test_points_y)
        Y = test_points[:,1].reshape(i_n_test_points_x, i_n_test_points_y)
        Y_Exact_Matrix = y_exact.reshape(i_n_test_points_x, i_n_test_points_y)
        save_contour_data(x = X, y = Y, z = Y_Exact_Matrix, output_path = i_output_path, filename= f"exact_solution", title = "Exact Solution")

    # ---------------------------------------------------------------#
    # ------------- PRE TRAINING INITIALISATIONS ------------------  #
    # ---------------------------------------------------------------#
    num_epochs = i_epochs  # num_epochs
    progress_bar = tqdm(total=num_epochs, desc='Training', unit='epoch', bar_format="{l_bar}{bar:40}{r_bar}{bar:-10b}", colour="green", ncols=100)
    loss_array = []   # total loss
    test_loss_array = [] # test loss
    time_array = []   # time per epoc
    # beta - boundary loss parameters
    beta = tf.constant(i_beta, dtype=i_dtype)

    inverse_params_array = []

    min_l1_error = np.inf

    # values to save 
    min_error_parameters  = {}
    min_error_arrays = {}
    # ---------------------------------------------------------------#
    # ------------- TRAINING LOOP ---------------------------------- #
    # ---------------------------------------------------------------#
    for epoch in range(num_epochs):   
        
        # Train the model
        batch_start_time = time.time()

        loss = model.train_step(beta=beta, bilinear_params_dict=bilinear_params_dict)
        
        elapsed = time.time() - batch_start_time
        progress_bar.update(1)
        # print(elapsed)
        time_array.append(elapsed)
        
        
        loss_array.append(loss['loss'])
        inverse_params_array.append(loss['inverse_params']['eps'].numpy())


        # ------ Progress bar update ------ #
        # if (epoch+1) % i_update_progress_bar == 0 or epoch == num_epochs-1:
        #     progress_bar.update(i_update_progress_bar)
        
        # ------ Intermediate results update ------ #
        if (epoch+1) % i_update_console_output == 0 or epoch == num_epochs-1:
            
            # Mean time per epoch
            mean_time = np.mean(time_array[-i_update_console_output:])

            #total time
            total_time_per_intermediate = np.sum(time_array[-i_update_console_output:])

            #epochs per second
            epochs_per_sec = i_update_console_output/np.sum(time_array[-i_update_console_output:])

            y_pred = model(test_points).numpy()
            y_pred = y_pred.reshape(-1,)

            # get errors
            l2_error, linf_error, l2_error_relative, linf_error_relative, \
                l1_error, l1_error_relative = compute_errors_combined(y_exact, y_pred)
            
            loss_pde = float(loss['loss_pde'].numpy())
            loss_dirichlet = float(loss['loss_dirichlet'].numpy())
            total_loss = float(loss['loss'].numpy())

            print(f"\nEpoch [bold]{epoch+1}/{num_epochs}[/bold]")
            print("[bold]--------------------[/bold]")
            print("[bold]Beta : [/bold]" , beta.numpy(), end=" ")
            print(f"[bold]Time/epoch : [/bold] {mean_time:.5f} s", end=" ")
            print("[bold]Epochs/second : [/bold]" , int(epochs_per_sec), end=" ")
            print(f"Learning Rate : {model.optimizer.lr.numpy():.3e}")
            print(f"Variational Losses || Pde Loss : [red]{loss_pde:.3e}[/red] Dirichlet Loss : [red]{loss_dirichlet:.3e}[/red] Total Loss : [red]{total_loss:.3e}[/red]")
            print(f"Test Losses        || L1 Error : {l1_error:.3e}", end=" ")
            print(f"L2 Error : {l2_error:.3e}", end=" ")
            print(f"Linf Error : {linf_error:.3e}", end="\n")
            # add inverse parameters and senor loss
            print(f"Inverse Parameters || eps : {loss['inverse_params']['eps'].numpy():.3e}", end=" ")
            print(f"Sensor Loss : { float(loss['sensor_loss'].numpy()):.3e}", end="\n")
            


            # append test loss
            test_loss_array.append(l1_error)
            

            # create a new array and perform cum_sum on time_array
            time_array_cum_sum = np.cumsum(time_array)
            
            #  Convert the three vectors into a single 2D matrix, where each vector is a column in the matrix
            if i_mesh_generation_method == "internal":
                # reshape y_pred into a 2D array
                y_pred = y_pred.reshape(i_n_test_points_x, i_n_test_points_y)
                
                #Error
                error = np.abs(Y_Exact_Matrix - y_pred)
                
                # plot the prediction
                save_contour_data(x = X, y = Y, z = y_pred, output_path = i_output_path, filename= f"prediction_{epoch+1}", title = "Prediction")
                # plot the error
                save_contour_data(x = X, y = Y, z = error, output_path = i_output_path, filename= f"error_{epoch+1}", title = "Error")


            elif i_mesh_generation_method == "external":
                solution_array = np.c_[y_pred, y_exact, np.abs(y_exact - y_pred)]
                domain.write_vtk(solution_array, output_path = i_output_path, filename= f"prediction_{epoch+1}.vtk", data_names = ["Sol","Exact", "Error"] )

        # check errors for  "last_test_error_epochs"
        if epoch >= num_epochs - i_last_test_error_epochs:
            
            # get prediction
            y_pred = model(test_points).numpy().reshape(-1,)

            # get errors
            l2_error, linf_error, l2_error_relative, linf_error_relative, \
                l1_error, l1_error_relative = compute_errors_combined(y_exact, y_pred)  
            
            
            if l1_error < min_l1_error:
                min_l1_error = l1_error
                min_error_parameters["epoch"] = epoch
                min_error_parameters["l1_error"] = l1_error
                min_error_parameters["l2_error"] = l2_error
                min_error_parameters["linf_error"] = linf_error
                min_error_parameters["l1_error_relative"] = l1_error_relative
                min_error_parameters["l2_error_relative"] = l2_error_relative
                min_error_parameters["linf_error_relative"] = linf_error_relative
                min_error_parameters["inverse_params"] = loss['inverse_params']['eps'].numpy()
                min_error_parameters["sensor_loss"] = float(loss['sensor_loss'].numpy())

                min_error_arrays["y_pred"] = y_pred
                min_error_arrays["error"] = np.abs(y_exact - y_pred)


    # Save the model
    model.save_weights(str(Path(i_output_path) / "model_weights_min_error"))

    # save all min error parameters in a file
    with open(str(Path(i_output_path) / "min_error_parameters.txt"), "w") as f:
        for key, value in min_error_parameters.items():
            f.write(f"{key} : {value}\n")
    
    # save min error arrays
    np.savetxt(str(Path(i_output_path) / "min_error_prediction.txt"), min_error_arrays["y_pred"])
    np.savetxt(str(Path(i_output_path) / "min_error_error.txt"), min_error_arrays["error"])         

    # close the progress bar
    progress_bar.close()

    # Save the model
    model.save_weights(str(Path(i_output_path) / "model_weights"))
    
    print("[INFO] Model Saved Successfully")

    
    # plot min error solution
    if i_mesh_generation_method == "internal":
        # reshape y_pred into a 2D array
        y_pred = y_pred.reshape(i_n_test_points_x, i_n_test_points_y)
        
        #Error
        error = np.abs(Y_Exact_Matrix - y_pred)
        
        # plot the prediction
        save_contour_data(x = X, y = Y, z = min_error_arrays["y_pred"], output_path = i_output_path, filename= f"final_prediction", title = "Final Prediction")
        # plot the error
        save_contour_data(x = X, y = Y, z = min_error_arrays["error"], output_path = i_output_path, filename= f"final_error", title = "Final Error")


    elif i_mesh_generation_method == "external":
        solution_array = np.c_[y_pred, y_exact, np.abs(y_exact - y_pred)]
        error = np.abs(y_exact - y_pred)
        domain.write_vtk(solution_array, output_path = i_output_path, filename= f"final_prediction.vtk", data_names = ["Sol","Exact", "Error"] )


    # domain.write_vtk(solution_array, output_path = i_output_path, filename= f"final_prediction.vtk", data_names = ["Sol","Exact", "Error"] )
    
    
    # print the Error values in table
    print_table_edge("Error Values", ["Error Type", "Value"], \
                ["L2 Error", "Linf Error", "Relative L2 Error", "Relative Linf Error", "L1 Error", "Relative L1 Error"], \
                [min_error_parameters["l2_error"], min_error_parameters["linf_error"], min_error_parameters["l2_error_relative"], \
                min_error_parameters["linf_error_relative"], min_error_parameters["l1_error"], min_error_parameters["l1_error_relative"]])

    # print the time values in table
    print_table_edge("Time Values", ["Time Type", "Value"], \
                ["Time per Epoch(s) - Median",   "Time per Epoch(s) IQR-25% ",   "Time per Epoch(s) IQR-75% ",  \
                 "Mean without first(s)", "Mean with first(s)" "Epochs per second" , "Total Train Time"], \
                [np.median(time_array[1:]), np.percentile(time_array[1:], 25),\
                np.percentile(time_array[1:], 75), 
                np.mean(time_array[1:]), np.mean(time_array),
                int(i_epochs/np.sum(time_array)) , np.sum(time_array[1:]) ])

    # save all the arrays as numpy arrays
    np.savetxt(str(Path(i_output_path) / "loss_function.txt"), np.array(loss_array))
    np.savetxt(str(Path(i_output_path) / "prediction.txt"), y_pred)
    np.savetxt(str(Path(i_output_path) / "exact.txt"), y_exact)
    np.savetxt(str(Path(i_output_path) / "error.txt"), error)
    np.savetxt(str(Path(i_output_path) / "time_per_epoch.txt"), np.array(time_array))
    np.savetxt(str(Path(i_output_path) / "inverse_params.txt"), np.array(inverse_params_array))

    # copy the input file to the output folder
    os.system(f"cp {sys.argv[1]} {i_output_path}")

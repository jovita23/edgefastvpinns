# Purpose: To save all the tensors of different ranks to be used in Edge Devices. 

# import Libraries
import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
from rich.progress import track
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn
from datetime import datetime
import yaml
import sys
import os
import time




os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # or any {'0', '1', '2'}

console = Console()

# import the FE classes for 2D
from src.FE_2D.basis_function_2d import *
from src.FE_2D.fespace2d import *

# Import the Geometry class
from src.Geometry.geometry_2d import *

# import the model class
from src.model.model_gear import *

# import the example file
from examples.gear import *

# import the data handler class
from src.data.datahandler2d import *

# import the plot utils
from src.utils.plot_utils import *
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
    i_seed  = config['experimentation']['seed']

    np.random.seed(i_seed)
    tf.random.set_seed(i_seed)
    tf.keras.utils.set_random_seed(i_seed)
    tf.config.experimental.enable_op_determinism()

    i_decomposition_type = config['decomposition_type']

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
    i_rank_list = config['fe']['rank_list']


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

    i_beta = config['pde']['beta']

    i_update_progress_bar = config['logging']['update_progress_bar']
    i_update_console_output = config['logging']['update_console_output']
    i_update_solution_images = config['logging']['update_solution_images']
    i_plot_residual_images = config['logging']['plot_residual_images']
    i_print_verbose = config['logging']['print_verbose']


    i_use_wandb = config['wandb']['use_wandb']
    i_wandb_project_name = config['wandb']['project_name']
    i_wandb_run_prefix = config['wandb']['wandb_run_prefix']
    i_wandb_entity = config['wandb']['entity']


    #i_update_console_output = config['logging']['update_console_output']
    i_test_errors_last_n_epochs = config['logging']['test_errors_last_n_epochs']
 
    min_l1_error = np.inf

    # values to save 
    min_error_parameters  = {}
    min_error_arrays = {}
    # ---------------------------------------------------------------#
    if i_decomposition_type == "tucker":

        from src.physics.cd2d_tucker import *
        from src.data.tucker_decom import decomposition

    elif i_decomposition_type == "cp":


        from src.physics.cd2d_cp import *    
        from src.data.cp_decom import decomposition

    elif i_decomposition_type == "ttd":


        from src.physics.cd2d_ttd import *    
        from src.data.ttd_decom import decomposition    
    
    #For expansion of GPU Memory
    if i_set_memory_growth:
        console.print("[INFO] Setting memory growth for GPU")
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
    # Commented out by thivin- Already saved as pkl file
    # fespace = Fespace2D(mesh = domain.mesh, cells=cells, boundary_points=boundary_points, 
    #                     cell_type=domain.mesh_type, fe_order=i_fe_order, fe_type =i_fe_type ,quad_order=i_quad_order, quad_type = i_quad_type, \
    #                     fe_transformation_type="bilinear", bound_function_dict = bound_function_dict, \
    #                     bound_condition_dict = bound_condition_dict, \
    #                     forcing_function=rhs, output_path=i_output_path)
    
    # save the fespace object as a binary file 
    fespace_path = "fespace_gear.pkl"
    
    # # save the fespace object using pickle into a binary file
    # import pickle
    # with open(fespace_path, 'wb') as f:
    #     pickle.dump(fespace, f)

    # read the fespace object from the binary file
    import pickle
    with open(fespace_path, 'rb') as f:
        fespace = pickle.load(f)
    

    # Instantiate the DataHandler2D class
    # datahandler = DataHandler2D(fespace, domain, dtype=i_dtype)

    # with open("datahandler.pkl", 'wb') as f:
    #     pickle.dump(datahandler, f)
    
    # read the datahandler object from the binary file
    with open("datahandler.pkl", 'rb') as f:
        datahandler = pickle.load(f)
    

    # get the input data for the PDE
    train_dirichlet_input, train_dirichlet_output = datahandler.get_dirichlet_input()

    # get bilinear parameters
    # this function will obtain the values of the bilinear parameters from the model
    # and convert them into tensors of desired dtype
    bilinear_params_dict = datahandler.get_bilinear_params_dict_as_tensors(get_bilinear_params_dict)

    start_time = time.time()
    print("Decomposing the tensors")
    core_x,factors_x = decomposition(datahandler.grad_x_mat_list,i_rank_list)
    core_y,factors_y = decomposition(datahandler.grad_y_mat_list,i_rank_list)
    
    end_time = time.time()
    print(f"Time taken for decomposition = {end_time - start_time} seconds")


    # Generate a Dict to serialise all the values and save it to a file
    save_dict = {}

    save_dict['time_taken'] = end_time - start_time
    save_dict['rank_list'] = i_rank_list
    save_dict['core_x'] = core_x
    save_dict['factors_x'] = factors_x
    save_dict['core_y'] = core_y
    save_dict['factors_y'] = factors_y
    save_dict['train_dirichlet_input'] = train_dirichlet_input
    save_dict['train_dirichlet_output'] = train_dirichlet_output
    save_dict['bilinear_params_dict'] = bilinear_params_dict
    save_dict['beta'] = i_beta
    save_dict['forcing_function_list'] = datahandler.forcing_function_list
    save_dict['bound_function_dict'] = bound_function_dict
    save_dict['bound_condition_dict'] = bound_condition_dict

    file_path = Path(f"{saved_tensors}/data/{i_decomposition_type}_{i_rank_list[0]}.npy")

    # read the the dictionary from the binary file
    save_dict = np.load(file_path)

    # create a dummy tensor for the shape value matrix
    # This is for backward compatibility, though these tensors are not used. the current model\
    # uses the last dimension of the tensor to reshape existing tensors, so we need to keep the\
    # shape of the tensor same as the previous version (only foe last dimension)
    shape_val_mat_list = tf.random.uniform(shape=(1,1,i_quad_order**2), dtype=i_dtype)
    grad_x_mat_list = tf.random.uniform(shape=(1,1,i_quad_order**2), dtype=i_dtype)
    grad_y_mat_list = tf.random.uniform(shape=(1,1,i_quad_order**2), dtype=i_dtype)

    # the saved core tensors are in numpy format, so we need to convert them to tensors
    save_dict['core_x'] = tf.convert_to_tensor(save_dict['core_x'], dtype=i_dtype)
    save_dict['factors_x'] = [tf.convert_to_tensor(f, dtype=i_dtype) for f in save_dict['factors_x']]
    save_dict['core_y'] = tf.convert_to_tensor(save_dict['core_y'], dtype=i_dtype)
    save_dict['factors_y'] = [tf.convert_to_tensor(f, dtype=i_dtype) for f in save_dict['factors_y']]

    # print the shapes of the tensors
    print("core_x shape = ", save_dict['core_x'].shape)
    print("core_y shape = ", save_dict['core_y'].shape)
    print("factors_x shapes = ", [f.shape for f in save_dict['factors_x']])
    print("factors_y shapes = ", [f.shape for f in save_dict['factors_y']])

    params_dict = {}
    params_dict['n_cells'] = fespace.n_cells

    model = DenseModel(
        layer_dims=i_model_architecture,
        learning_rate_dict=i_learning_rate_dict,
        params_dict=params_dict,
        loss_function=pde_loss_cd2d,
        input_tensors_list=[datahandler.x_pde_list, train_dirichlet_input, train_dirichlet_output],
        orig_factor_matrices=[
            datahandler.shape_val_mat_list,
            datahandler.grad_x_mat_list,
            datahandler.grad_y_mat_list,save_dict['core_x'],save_dict['factor_x'],save_dict['core_y'],save_dict['factor_y']
        ],
        force_function_list=datahandler.forcing_function_list,
        tensor_dtype=i_dtype,
        use_attention=i_use_attention,
        activation=i_activation,
        hessian=False,
    )

    time_taken = []

    # train the model
    for epoch in range(i_epochs):
        start_time = time.time()
        loss = model.train_step(beta=beta, bilinear_params_dict=bilinear_params_dict)
        end_time = time.time()
        time_taken.append(end_time - start_time)

    # print these values as a table
    # Num epochs, Total time, Mean time per epoch, mean time per epoch, std dev of time per epoch, 25th percentile, 75th percentile

    time_taken = np.array(time_taken)
    total_time = np.sum(time_taken)
    mean_time_per_epoch = np.mean(time_taken)
    std_dev_time_per_epoch = np.std(time_taken)
    percentile_25 = np.percentile(time_taken, 25)
    percentile_75 = np.percentile(time_taken, 75)

    print("Total time = ", total_time)
    print("Mean time per epoch = ", mean_time_per_epoch)
    print("Std dev time per epoch = ", std_dev_time_per_epoch)
    print("25th percentile = ", percentile_25)
    print("75th percentile = ", percentile_75)

    time_dict = {}
    time_dict['total_time'] = total_time
    time_dict['mean_time_per_epoch'] = mean_time_per_epoch
    time_dict['std_dev_time_per_epoch'] = std_dev_time_per_epoch
    time_dict['percentile_25'] = percentile_25
    time_dict['percentile_75'] = percentile_75

    shape_dict = {}
    shape_dict['core_x_shape'] = save_dict['core_x'].shape
    shape_dict['core_y_shape'] = save_dict['core_y'].shape
    shape_dict['factors_x_shapes'] = [f.shape for f in save_dict['factors_x']]
    shape_dict['factors_y_shapes'] = [f.shape for f in save_dict['factors_y']]
    shape_dict['train_dirichlet_input_shape'] = train_dirichlet_input.shape
    shape_dict['train_dirichlet_output_shape'] = train_dirichlet_output.shape
    shape_dict['x_pde_list_shape'] = datahandler.x_pde_list.shape

    # save the time_dict to a file
    file_path = i_output_path / f"time_dict_{i_decomposition_type}_{i_rank_list[0]}.npy"
    np.save(file_path, time_dict)

    # save the shape_dict to a file
    file_path = i_output_path / f"shape_dict_{i_decomposition_type}_{i_rank_list[0]}.npy"
    np.save(file_path, shape_dict)


    

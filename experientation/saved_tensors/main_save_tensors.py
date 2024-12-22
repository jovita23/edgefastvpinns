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

    # save this as a binary file 
    save_path = folder / f"tensors_{i_decomposition_type}_{i_rank_list[0]}.npy"
    np.save(save_path, save_dict)

    console.print(f"[INFO] Saved the tensors to {save_path}")
    

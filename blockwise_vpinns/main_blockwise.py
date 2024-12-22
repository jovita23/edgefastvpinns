# Main File for running the Python code
# of all the cells within the given mesh
# This code is meant for execution of the FastVPINNs algorithm
# on an edge device with blockwise implementation

# Authors: Jovita






# import Libraries
import numpy as np
import tensorflow as tf
from pathlib import Path
#import t3f

import tensorly as tl


import yaml
import sys
import os
import time
import wandb

np.random.seed(1234)
tf.random.set_seed(1234)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # or any {'0', '1', '2'}


# import the FE classes for 2D
from src.FE_2D.basis_function_2d import *
from src.FE_2D.fespace2d import *

# Import the Geometry class
from src.Geometry.geometry_2d import *

# import the model class
from model_gear_blockwise import *

# import the example file
from examples.gear import *

#import physics for custom loss function
#from src.physics.tucker_new import *
#from src.physics.poisson2d import *

#from src.data.tucker_decom import decomposition
# import the data handler class
from src.data.datahandler_block import *

# import the plot utils
# from src.utils.plot_utils import *
from src.utils.compute_utils import *
from src.utils.print_utils import *
from src.utils.edge_utils import *

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

    i_decomposition_type = config['decomposition_type']

    # Extract geometry parameters
    i_mesh_generation_method = config['geometry']['mesh_generation_method']
    i_generate_mesh_plot = config['geometry']['generate_mesh_plot']
    i_mesh_type = config['geometry']['mesh_type']
    i_x_min = config['geometry']['internal_mesh_params']['x_min']
    i_x_max = config['geometry']['internal_mesh_params']['x_max']
    i_y_min = config['geometry']['internal_mesh_params']['y_min']
    i_y_max = config['geometry']['internal_mesh_params']['y_max']
    i_n_cells_x = config['geometry']['internal_mesh_params']['n_cells_x']
    i_n_cells_y = config['geometry']['internal_mesh_params']['n_cells_y']
    
    
    i_num_blocks  = config['geometry']['num_blocks']

    


    i_n_boundary_points = config['geometry']['internal_mesh_params']['n_boundary_points']
    i_n_test_points_x = config['geometry']['internal_mesh_params']['n_test_points_x']
    i_n_test_points_y = config['geometry']['internal_mesh_params']['n_test_points_y']
    i_mesh_file_name = config['geometry']['external_mesh_params']['mesh_file_name']
    i_boundary_refinement_level = config['geometry']['external_mesh_params']['boundary_refinement_level']
    i_boundary_sampling_method = config['geometry']['external_mesh_params']['boundary_sampling_method']
    
    # Extract the FE parameters
    i_fe_order = config['fe']['fe_order']
    i_fe_type = config['fe']['fe_type']
    i_quad_order = config['fe']['quad_order']
    i_quad_type = config['fe']['quad_type']
    i_rank_list = config['fe']['rank_list']

    i_is_matplotlib_available = config['is_matplotlib_available']
    if i_is_matplotlib_available:
        from src.utils.plot_utils import *


    # Extract the model parameters
    i_model_architecture = config['model']['model_architecture']
    i_activation = config['model']['activation']
    i_use_attention = config['model']['use_attention']
    i_epochs = config['model']['epochs']
    i_set_memory_growth = config['model']['set_memory_growth']
    i_learning_rate_dict = config['model']['learning_rate']
    i_dtype = config['model']['dtype']
    
    if i_dtype == "float64":
        i_dtype = tf.float64
    elif i_dtype == "float32":
        i_dtype = tf.float32
    else:
        print("[ERROR] The given dtype is not a valid tensorflow dtype")
        raise ValueError("The given dtype is not a valid tensorflow dtype")
    
    # Extract PDE Loss parameters
    i_beta = config['pde']['beta']

    # Extract the logging parameters
    i_update_progress_bar = config['logging']['update_progress_bar']
    i_update_console_output = config['logging']['update_console_output']
    i_update_solution_images = config['logging']['update_solution_images']
    i_plot_residual_images = config['logging']['plot_residual_images']
    i_print_verbose = config['logging']['print_verbose']


    i_use_wandb = config['wandb']['use_wandb']
    i_wandb_project_name = config['wandb']['project_name']
    i_wandb_run_prefix = config['wandb']['wandb_run_prefix']
    i_wandb_entity = config['wandb']['entity']


    i_update_console_output = config['logging']['update_console_output']
    i_test_errors_last_n_epochs = config['logging']['test_errors_last_n_epochs']
 

    min_l1_error = np.inf

    # values to save 
    min_error_parameters  = {}
    min_error_arrays = {}

    if i_decomposition_type == "tucker":

        from src.physics.cd2d_tucker import *
       
    elif i_decomposition_type == "cp":


        from src.physics.cd2d_cp import *    
        

    elif i_decomposition_type == "ttd":


        from src.physics.cd2d_ttd import *    
        
    # ---------------------------------------------------------------#
    # Initialise wandb
    if i_use_wandb:
        run_name = i_wandb_run_prefix
        wandb.init(project=i_wandb_project_name, entity=i_wandb_entity, name=run_name, config=config)
    
    # For dynamic allocation of GPU Memory
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


    # Read mesh from a .mesh file if mesh_generation_method is external
    if i_mesh_generation_method == "external":
        cells, boundary_points = domain.read_mesh(mesh_file = i_mesh_file_name, \
                                                boundary_point_refinement_level=i_boundary_refinement_level, \
                                                bd_sampling_method=i_boundary_sampling_method, \
                                                refinement_level=0)

    # Generate a uniform quad mesh if mesh_generation_method is internal
    elif i_mesh_generation_method == "internal":
        cells, boundary_points = domain.generate_quad_mesh_internal(x_limits = [i_x_min, i_x_max], \
                                                                    y_limits = [i_y_min, i_y_max], \
                                                                    n_cells_x =  i_n_cells_x, \
                                                                    n_cells_y = i_n_cells_y, \
                                                                    num_boundary_points=i_n_boundary_points)


    
    #get the boundary function dictionary from example file
    bound_function_dict, bound_condition_dict = get_boundary_function_dict(), get_bound_cond_dict()
    
    
    # get fespace2d
    fespace = Fespace2D(mesh = domain.mesh, cells = cells, boundary_points=boundary_points, 
                        cell_type=domain.mesh_type, fe_order=i_fe_order, fe_type =i_fe_type ,quad_order=i_quad_order, quad_type = i_quad_type, \
                        fe_transformation_type="bilinear", bound_function_dict = bound_function_dict, \
                        bound_condition_dict = bound_condition_dict, \
                        forcing_function=rhs, output_path=i_output_path, generate_mesh_plot = i_generate_mesh_plot)


    # Instantiate the DataHandler2D class
    datahandler = DataHandler2D(fespace, domain, i_num_blocks,i_decomposition_type, i_rank_list, dtype=i_dtype)

    

    

    
    params_dict = {}
    params_dict['n_cells'] = fespace.n_cells    

    # get the input data for the PDE
    train_dirichlet_input, train_dirichlet_output = datahandler.get_dirichlet_input()
    bilinear_params_dict = datahandler.get_bilinear_params_dict_as_tensors(get_bilinear_params_dict)

    


    model = DenseModelEdge(layer_dims = i_model_architecture, learning_rate_dict = i_learning_rate_dict, \
                       params_dict = params_dict, \
                       loss_function = pde_loss_cd2d, input_tensors_list = [datahandler.x_pde_blocks, train_dirichlet_input, train_dirichlet_output], \
                        orig_factor_matrices = [datahandler.shape_val_mat_blocks , datahandler.grad_x_mat_blocks, datahandler.grad_y_mat_blocks], \
                        force_function_list = datahandler.forcing_function_blocks, \
                        tensor_dtype = i_dtype,
                        activation=i_activation, \
                        hessian=False)
    


    ## ------------- Need to do the below to print the summary of the custom model -------- ##

    # ---------------------------------------------------------------#
    # --------------    Get Testing points   ----------------------- #
    # ---------------------------------------------------------------#
    
    # test_points = np.c_[xx.ravel(), yy.ravel()]
    #code obtains the test points based on internal or external mesh

    

    test_points = domain.get_test_points()
    # print(f"Number of Test Points = {test_points.shape[0]}")
    # y_exact = exact_solution(test_points[:,0], test_points[:,1])

    if i_mesh_generation_method == "external":
        exact_db = pd.read_csv("fem_output_gear_forward_sin.csv", header=None, delimiter=",")
        y_exact = exact_db.iloc[:,2].values.reshape(-1,)

    else:
        y_exact = exact_solution(test_points[:,0], test_points[:,1])
    

    # save points for plotting
    if i_mesh_generation_method == "internal":
        X = test_points[:,0].reshape(i_n_test_points_x, i_n_test_points_y)
        Y = test_points[:,1].reshape(i_n_test_points_x, i_n_test_points_y)
        Y_Exact_Matrix = y_exact.reshape(i_n_test_points_x, i_n_test_points_y)

        # plot the exact solution
        save_contour_data(x = X, y = Y, z = Y_Exact_Matrix, output_path = i_output_path, filename= "exact_solution")
    

    # ---------------------------------------------------------------#
    # ------------- PRE TRAINING INITIALISATIONS ------------------  #
    # ---------------------------------------------------------------#
    num_epochs = i_epochs  # num_epochs

    loss_array = []   # total loss

    test_loss_array = [] # test loss

    time_array = []   # time per epoch

    # beta - boundary loss parameters
    beta = tf.constant(i_beta, dtype=i_dtype)



    # ---------------------------------------------------------------#
    # ------------- TRAINING LOOP ---------------------------------- #
    # ---------------------------------------------------------------#
    for epoch in range(num_epochs):   
        
        # Train the model
        batch_start_time = time.time()
        loss = model.train_step(beta=beta,num_blocks = i_num_blocks, bilinear_params_dict = bilinear_params_dict)

        
        elapsed = time.time() - batch_start_time
        time_array.append(elapsed)
                
        loss_array.append(loss['loss'])
        

       
        
        # ------ Intermediate results update ------ #
        if (epoch+1) % i_update_console_output == 0 or epoch == num_epochs-1:
            
            print(f"\nEpoch {epoch+1} ")
            print(f"{'-*'*int(((epoch+1)/num_epochs)*10)}")
            

  

            y_pred = model(test_points).numpy()
            y_pred = y_pred.reshape(-1,)

            # get errors
            l2_error, linf_error, l2_error_relative, linf_error_relative, \
                l1_error, l1_error_relative = compute_errors_combined(y_exact, y_pred)
            
            loss_pde = float(loss['loss_pde'].numpy())
            loss_dirichlet = float(loss['loss_dirichlet'].numpy())
            total_loss = float(loss['loss'].numpy())

            if i_print_verbose:
                print(f"\nEpoch {epoch+1}/{num_epochs}")
                print("--------------------")
                print("Beta : " , beta.numpy(), end=" ")
                print(f"Learning Rate : {model.optimizer.lr.numpy():.3e}")
                print(f"Variational Losses || Pde Loss : {loss_pde:.3e} Dirichlet Loss : {loss_dirichlet:.3e} Total Loss : {total_loss:.3e}")
                print(f"Test Losses        || L1 Error : {l1_error:.3e}", end=" ")
                print(f"L2 Error : {l2_error:.3e}", end=" ")
                print(f"Linf Error : {linf_error:.3e}", end="\n")


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

                if i_is_matplotlib_available:
                        plot_contour(X, Y, y_pred.reshape(i_n_test_points_x, i_n_test_points_y), i_output_path, f"predicted_solution_{epoch+1}", "Predicted Solution")
                        plot_contour(X, Y, error.reshape(i_n_test_points_x, i_n_test_points_y), i_output_path, f"error_{epoch+1}", "Error")



                
                # # # plot the prediction
                # save_contour_data(x = X, y = Y, z = y_pred, output_path = i_output_path, filename= f"prediction_{epoch+1}")

                # # # plot the error
                # save_contour_data(x = X, y = Y, z = error, output_path = i_output_path, filename= f"error_{epoch+1}")


            elif i_mesh_generation_method == "external":
                solution_array = np.c_[y_pred, y_exact, np.abs(y_exact - y_pred)]
                domain.write_vtk(solution_array, output_path = i_output_path, filename= f"prediction_{epoch+1}.vtk", data_names = ["Sol","Exact", "Error"] )

            
            # if i_use_wandb:
            #     wandb.log({"epoch": epoch+1, "loss_pde": loss_pde, "loss_dirichlet": loss_dirichlet, "total_loss": total_loss, "time_per_epoch": mean_time, "epochs_per_sec": epochs_per_sec, \
            #     "l2": l2_error, "linf": linf_error, "l2_relative": l2_error_relative, "linf_relative": linf_error_relative})
        if epoch >= i_epochs - i_test_errors_last_n_epochs:



            y_pred = model(test_points).numpy().flatten()
            error  = np.abs(y_exact - y_pred)

            # compute Errors
            l2_error, linf_error, l2_error_relative, linf_error_relative, \
                l1_error, l1_error_relative = compute_errors_combined(y_exact, y_pred)

            if l1_error < min_l1_error:
                min_l1_error = l1_error
                min_error_parameters['epoch'] = epoch
                min_error_parameters['l1_error'] = l1_error
                min_error_parameters['l1_error_relative'] = l1_error_relative
                min_error_parameters['l2_error'] = l2_error
                min_error_parameters['l2_error_relative'] = l2_error_relative
                min_error_parameters['linf_error'] = linf_error
                min_error_parameters['linf_error_relative'] = linf_error_relative
                min_error_arrays['error'] = error
                min_error_arrays['y_pred'] = y_pred

    plot_loss_function(loss_array,output_path=i_output_path)            

    save_loss_function(loss_array, i_output_path, "loss_function.txt")  # saves NN loss
    save_test_loss_function(test_loss_array, i_output_path) # saves test loss
    # Save the model
    model.save_weights(str(Path(i_output_path) / "model_weights"))
    #model.save(str(Path(i_output_path) / "model_weights"))



    # if i_use_wandb:
    #     # get all files, which start with the name "model_weights"
    #     model_weights = [str(Path(i_output_path) / i) for i in os.listdir(i_output_path) if i.startswith("model_weights")]
    #     # save all the model weights
    #     for i in model_weights:
    #         wandb.save(i)

    #     wandb.save(str(Path(i_output_path) / "model_weights"))
    
    # print("[INFO] Model Saved Successfully")

    # predict the values
    y_pred = model(test_points).numpy().reshape(-1,)

    # plot the loss function, prediction, exact solution and error
    
    # get errors
    l2_error, linf_error, l2_error_relative, linf_error_relative, \
                l1_error, l1_error_relative = compute_errors_combined(y_exact, y_pred)
    

    solution_array = np.c_[y_pred, y_exact, np.abs(y_exact - y_pred)]
    
    if i_mesh_generation_method == "internal":
        # reshape y_pred into a 2D array
        y_pred = y_pred.reshape(i_n_test_points_x, i_n_test_points_y)
        
        #Error
        error = np.abs(Y_Exact_Matrix - y_pred)
        
        # plot the prediction
        save_contour_data(x = X, y = Y, z = y_pred, output_path = i_output_path, filename= f"final_prediction")
        # plot the error
        save_contour_data(x = X, y = Y, z = error, output_path = i_output_path, filename= f"final_error")


    elif i_mesh_generation_method == "external":
        solution_array = np.c_[y_pred, y_exact, np.abs(y_exact - y_pred)]
        error = np.abs(y_exact - y_pred)
        domain.write_vtk(solution_array, output_path = i_output_path, filename= f"final_prediction.vtk", data_names = ["Sol","Exact", "Error"] )


    # domain.write_vtk(solution_array, output_path = i_output_path, filename= f"final_prediction.vtk", data_names = ["Sol","Exact", "Error"] )


    if i_is_matplotlib_available:
        plot_contour(X, Y, min_error_arrays["y_pred"].reshape(i_n_test_points_x, i_n_test_points_y), i_output_path, "min_error_predicted_solution", "Predicted Solution")
        plot_contour(X, Y, y_exact.reshape(i_n_test_points_x, i_n_test_points_y), i_output_path, "min_error_exact_solution", "Exact Solution")
        plot_contour(X, Y, min_error_arrays["error"].reshape(i_n_test_points_x, i_n_test_points_y), i_output_path, "min_error_error", "Error")
    
    
    # print the Error values in table
    print("\nError Values")
    print("------------")
    print(f"L1 Error : {l1_error:.3e}")
    print(f"L2 Error : {l2_error:.3e}")
    print(f"Linf Error : {linf_error:.3e}")
    print(f"Relative L1 Error : {l1_error_relative:.3e}")
    print(f"Relative L2 Error : {l2_error_relative:.3e}")
    print(f"Relative Linf Error : {linf_error_relative:.3e}")
    print("\n")
    

    print("Error Values")
    print("-----------------")
    print(f"Minimum L1 Error            :  {min_error_parameters['l1_error']:.3e}")
    print(f"Minimum L1 Relative Error   :  {min_error_parameters['l1_error_relative']:.3e}")
    print(f"Minimum L2 Error            :  {min_error_parameters['l2_error']:.3e}")
    print(f"Minimum L2 Relative Error   :  {min_error_parameters['l2_error_relative']:.3e}")
    print(f"Minimum Linf Error          :  {min_error_parameters['linf_error']:.3e}")
    print(f"Minimum Linf Relative Error :  {min_error_parameters['linf_error_relative']:.3e}")
    print("\n")

    print("Time Values")
    print("------------")
    print(f"Time per Epoch(s) - Median : {np.median(time_array):.3e}")
    print(f"Epochs per second : {int(i_epochs/np.sum(time_array))}")
    print(f"Total Train Time : {np.sum(time_array):.3e}")


    # save all the arrays as numpy arrays
    np.savetxt(str(Path(i_output_path) / "loss_function.txt"), np.array(loss_array))
    np.savetxt(str(Path(i_output_path) / "prediction.txt"), y_pred)
    np.savetxt(str(Path(i_output_path) / "exact.txt"), y_exact)
    np.savetxt(str(Path(i_output_path) / "error.txt"), error)
    np.savetxt(str(Path(i_output_path) / "time_per_epoch.txt"), np.array(time_array))


    if i_use_wandb:
        wandb.log({"l2": l2_error, "linf": linf_error, "l2_relative": l2_error_relative, "linf_relative": linf_error_relative, "l1": l1_error, "l1_relative": l1_error_relative})
        
    #     wandb.log({"loss_function": wandb.Image(i_output_path + "/loss_function.png")})
        wandb.save(sys.argv[1])

    #     # save the numpy arrays
        wandb.save(str(Path(i_output_path) / "loss_function.txt"))
        wandb.save(str(Path(i_output_path) / "prediction.txt"))
        wandb.save(str(Path(i_output_path) / "exact.txt"))
        wandb.save(str(Path(i_output_path) / "error.txt"))
        wandb.save(str(Path(i_output_path) / "time_per_epoch.txt"))

    #     if i_mesh_generation_method == "internal":
    #         wandb.save(str(Path(i_output_path) / "final_prediction.png"))
    #         wandb.save(str(Path(i_output_path) / "final_error.png"))
    #     elif i_mesh_generation_method == "external":
    #         wandb.save(str(Path(i_output_path) / "final_prediction.vtk"))
    #     else:
    #         pass


    # copy the input file to the output folder
    os.system(f"cp {sys.argv[1]} {i_output_path}")

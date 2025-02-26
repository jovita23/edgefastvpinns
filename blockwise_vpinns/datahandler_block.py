
# @Author : Jovita Biju


#The blockwise assembling of tensor is handled by this class

from src.FE_2D.fespace2d import *
from src.Geometry.geometry_2d import *
from src.data.tensor_decom import decomposition
import tensorflow as tf

class DataHandler2D():
    """
    This class is to handle data for 2D problems, convert them into tensors using custom tf functions
    Responsible for all type conversions and data handling
    Note: All inputs to these functions are generally numpy arrays with dtype np.float64
            So we can either maintain the same dtype or convert them to tf.float32 ( for faster computation )
    
    Attributes:
    - fespace (FESpace2D): The FESpace2D object
    - domain (Domain2D): The Domain2D object
    - shape_val_mat_list (list): List of shape function values for each cell
    - grad_x_mat_list (list): List of shape function derivatives with respect to x for each cell
    - grad_y_mat_list (list): List of shape function derivatives with respect to y for each cell
    - x_pde_list (list): List of actual coordinates of the quadrature points for each cell
    - forcing_function_list (list): List of forcing function values for each cell
    - dtype (tf.DType): The tensorflow dtype to be used for all the tensors

    Methods:
    - get_pde_input(): Returns the input for the PDE training data
    - get_dirichlet_input(): Returns the input for the Dirichlet boundary data
    - get_test_points(num_test_points): Returns the test points
    - get_bilinear_params_dict_as_tensors(function): Accepts a function from example file and converts all the values into tensors of the given dtype
    - get_sensor_data(exact_sol, num_sensor_points, mesh_type, file_name=None): Returns the sensor data
    - get_inverse_params(inverse_params_dict_function): Accepts a function from example file and converts all the values into tensors of the given dtype

    """
    
    def __init__(self, fespace, domain, num_blocks,decompsition_type, rank_list, dtype):
        """
        Constructor for the DataHandler2D class

        Parameters:
        - fespace (FESpace2D): The FESpace2D object
        - domain (Domain2D): The Domain2D object
        - dtype (tf.DType): The tensorflow dtype to be used for all the tensors

        Returns:
        None
        """

        self.fespace = fespace
        self.domain = domain
        self.num_blocks = num_blocks
        self.shape_val_mat_blocks = []
        self.grad_x_mat_blocks = []
        self.grad_y_mat_blocks = []
        self.x_pde_blocks = []
        self.forcing_function_blocks = []
        self.dtype = dtype

        self.decomposition_type = decompsition_type
        
        self.rank_list = rank_list

        total_cells = fespace.n_cells
        cells_per_block = total_cells // num_blocks

        remainder = total_cells % num_blocks
        start_cell = 0


        # check if the given dtype is a valid tensorflow dtype
        if not isinstance(self.dtype, tf.DType):
            raise TypeError("The given dtype is not a valid tensorflow dtype")
        
        for block_idx in range(num_blocks):
            num_cells_in_block = cells_per_block + (1 if block_idx < remainder else 0)
            end_cell = start_cell + num_cells_in_block

            shape_val_mat_block = []
            grad_x_mat_block = []
            grad_y_mat_block = []
            x_pde_block = []
            forcing_function_block = []

            for cell_index in range(start_cell, end_cell):
                shape_val_mat = tf.constant(self.fespace.get_shape_function_val(cell_index), dtype=self.dtype)
                grad_x_mat = tf.constant(self.fespace.get_shape_function_grad_x(cell_index), dtype=self.dtype)
                grad_y_mat = tf.constant(self.fespace.get_shape_function_grad_y(cell_index), dtype=self.dtype)
                x_pde = tf.constant(self.fespace.get_quadrature_actual_coordinates(cell_index), dtype=self.dtype)
                forcing_function = tf.constant(self.fespace.get_forcing_function_values(cell_index), dtype=self.dtype)

                shape_val_mat_block.append(shape_val_mat)
                grad_x_mat_block.append(grad_x_mat)
                grad_y_mat_block.append(grad_y_mat)
                x_pde_block.append(x_pde)
                forcing_function_block.append(forcing_function)

           
            shape_val_mat_block = tf.stack(shape_val_mat_block, axis=0)
            grad_x_mat_block = tf.stack(grad_x_mat_block, axis=0)
            grad_y_mat_block = tf.stack(grad_y_mat_block, axis=0)
            x_pde_block = tf.reshape(x_pde_block, [-1, 2])
            forcing_function_block = tf.concat(forcing_function_block, axis=1)
            
            #shape_val_mat_block = [decomposition(shape_val_mat_block, rank_list)]
            grad_x_mat_block_decomposed = list(decomposition(grad_x_mat_block, rank_list,type=decompsition_type))
            grad_y_mat_block_decomposed = list(decomposition(grad_y_mat_block, rank_list,type=decompsition_type))

            #free up the memory by deleting the original tensors
            del grad_x_mat_block
            del grad_y_mat_block

            self.shape_val_mat_blocks.append(shape_val_mat_block)
            self.grad_x_mat_blocks.append(grad_x_mat_block_decomposed)
            self.grad_y_mat_blocks.append(grad_y_mat_block_decomposed)
            self.x_pde_blocks.append(x_pde_block)
            self.forcing_function_blocks.append(forcing_function_block)

            start_cell = end_cell 





    def get_pde_input(self):
        """
        This function will return the input for the PDE training data

        Returns:
        - input_pde (tf.Tensor): The input for the PDE training data
        """
        return self.fespace.get_pde_training_data()
    
    def get_dirichlet_input(self):
        """
        This function will return the input for the Dirichlet boundary data

        Args:
        None

        Returns:
        - input_dirichlet (tf.Tensor): The input for the Dirichlet boundary data
        - actual_dirichlet (tf.Tensor): The actual Dirichlet boundary data

        """
        input_dirichlet, actual_dirichlet = self.fespace.generate_dirichlet_boundary_data()
        
        # convert to tensors
        input_dirichlet = tf.constant(input_dirichlet, dtype=self.dtype)
        actual_dirichlet = tf.constant(actual_dirichlet, dtype=self.dtype)
        actual_dirichlet = tf.reshape(actual_dirichlet, [-1, 1])

        return input_dirichlet, actual_dirichlet

    def get_test_points(self, num_test_points):
        """
        This function will return the test points for the given domain

        Args:
        - num_test_points (int): The number of test points to be generated

        Returns:
        - test_points (tf.Tensor): The test points for the given domain
        """
        

        self.test_points = self.domain.generate_test_points(num_test_points)
        self.test_points = tf.constant(self.test_points, dtype=self.dtype)
        return self.test_points

    def get_bilinear_params_dict_as_tensors(self, function):
        """
        Accepts a function from example file and converts all the values into tensors of the given dtype

        Parameters:
        - function (function): The function from the example file which returns the bilinear parameters dictionary

        Returns:
        - bilinear_params_dict (dict): The bilinear parameters dictionary with all the values converted to tensors
        """
        
        # get the dictionary of bilinear parameters
        bilinear_params_dict = function()

        # loop over all keys and convert the values to tensors
        for key in bilinear_params_dict.keys():
            bilinear_params_dict[key] = tf.constant(bilinear_params_dict[key], dtype=self.dtype)
        
        return bilinear_params_dict
    

    # to be used only in inverse problems
    def get_sensor_data(self, exact_sol, num_sensor_points, mesh_type, file_name=None):
        """
        Accepts a function from example file and converts all the values into tensors of the given dtype

        Parameters:
        - exact_sol (function): The function from the example file which returns the exact solution
        - num_sensor_points (int): The number of sensor points to be generated
        - mesh_type (str): The type of mesh to be used for sensor data generation
        - file_name (str): The name of the file to be used for external mesh generation

        Returns:
        - points (tf.Tensor): The sensor points
        - sensor_values (tf.Tensor): The sensor values
        """
        print(f"mesh_type = {mesh_type}")
        if (mesh_type == "internal"):
            # Call the method to get the sensor data
            points, sensor_values = self.fespace.get_sensor_data(exact_sol, num_sensor_points)
        elif (mesh_type == "external"):
            # Call the method to get the sensor data
            points, sensor_values = self.fespace.get_sensor_data_external(exact_sol, num_sensor_points, file_name)
        # convert the points and sensor values into tensors
        points = tf.constant(points, dtype=self.dtype)
        sensor_values = tf.constant(sensor_values, dtype=self.dtype)

        sensor_values = tf.reshape(sensor_values, [-1, 1])
        points = tf.reshape(points, [-1, 2])



        return points, sensor_values
    

    # get inverse param dict as tensors
    def get_inverse_params(self, inverse_params_dict_function):
        """
        Accepts a function from example file and converts all the values into tensors of the given dtype

        Parameters:
        - inverse_params_dict_function (function): The function from the example file which returns the inverse parameters dictionary

        Returns:
        - inverse_params_dict (dict): The inverse parameters dictionary with all the values converted to tensors
        """
        # loop over all keys and convert the values to tensors

        inverse_params_dict = inverse_params_dict_function()

        for key in inverse_params_dict.keys():
            inverse_params_dict[key] = tf.constant(inverse_params_dict[key], dtype=self.dtype)
        
        return inverse_params_dict
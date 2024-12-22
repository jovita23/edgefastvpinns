import numpy as np
# import meshio
# import pygmsh
from pathlib import Path
# from pyDOE import lhs

import gmsh
import meshio

import pyvista as pv
import tensorflow as tf

# import print_utils
# from src.utils.print_utils import print_table, print_table_multicolumns

# Start a virtual display buffer
pv.start_xvfb()



class Geometry_2D():
    """
    Defines functions to read mesh from Gmsh and internal mesh for 2D problems.

    Attributes:
    - mesh_type (str): The type of mesh to be used.
    - mesh_generation_method (str): The method used to generate the mesh.
    - n_test_points_x (int): The number of test points in the x-direction.
    - n_test_points_y (int): The number of test points in the y-direction.
    - output_folder (str): The path to the output folder.

    Methods:
    - read_mesh: Reads the mesh from a Gmsh .msh file and extracts cell information.
    - generate_quad_mesh_internal: Generates and saves a quadrilateral mesh with physical curves.
    - get_test_points: Returns the test points.
    - write_vtk: Writes the data to a VTK file.
    - save_vtk_as_image: Saves the VTK file as an image.

    """
    
    def __init__(self, mesh_type: str, mesh_generation_method: str, n_test_points_x: int, n_test_points_y: int, output_folder: str):
        """
        Constructor for Geometry_2D class.
        """
        self.mesh_type = mesh_type
        self.mesh_generation_method = mesh_generation_method
        self.n_test_points_x = n_test_points_x
        self.n_test_points_y = n_test_points_y
        self.output_folder = output_folder

        # To be filled - only when mesh is internal
        self.n_cells_x = None
        self.n_cells_y = None
        self.x_limits = None
        self.y_limits = None


        # to be filled by external 
        self.mesh_file_name = None

        
    
    def read_mesh(self, mesh_file: str, boundary_point_refinement_level: int, bd_sampling_method: str, refinement_level: int):
        """
        Reads mesh from a Gmsh .msh file and extracts cell information.

        Parameters:
        mesh_file (str): The path to the mesh file.
        boundary_point_refinement_level (int): The number of boundary points to be generated.
        bd_sampling_method (str): The method used to generate the boundary points.
        refinement_level (int): The number of times the mesh should be refined.

        Returns:
        cell_points (numpy.ndarray): The cell points.
        bd_dict (dict): The dictionary of boundary points.
        """
    
        self.mesh_file_name = mesh_file

        bd_sampling_method = "uniform" # "uniform" or "lhs"

        file_extension = Path(mesh_file).suffix
            
        if(file_extension != '.mesh'):
            raise ValueError('Mesh file should be in .mesh format.')

        # Read mesh using meshio
        self.mesh = meshio.read(mesh_file)

        if(self.mesh_type == "quadrilateral"):
            # Extract cell information
            cells = self.mesh.cells_dict["quad"]
        elif(self.mesh_type == "triangle"):
            cells = self.mesh.cells_dict["triangle"]

        else:
            raise ValueError('Mesh type should be either quadrilateral or triangle.')
            
        num_cells = cells.shape[0]
        print(f"[INFO] : Number of cells = {num_cells}")
        cell_points = self.mesh.points[cells][:,:,0:2] # remove the z coordinate, which is 0 for all points

        # loop over all cells and rearrange the points in anticlockwise direction
        for i in range(num_cells):
            cell = cell_points[i]
            # get the centroid of the cell
            centroid = np.mean(cell,axis=0)
            # get the angle of each point with respect to the centroid
            angles = np.arctan2(cell[:,1] - centroid[1], cell[:,0] - centroid[0])
            # sort the points based on the angles
            cell_points[i] = cell[np.argsort(angles)]

        # Extract number of points within each cell
        num_points_per_cell = cells.shape[1]
        print(f"[INFO] : Number of points per cell = {cell_points.shape}")
        

        # Collect the Boundary point id's within the domain
        boundary_edges = self.mesh.cells_dict["line"]

        # Using the point id, collect the coordinates of the boundary points
        boundary_coordinates = self.mesh.points[boundary_edges]
        
        # Number of Existing Boundary points
        print(f"[INFO] : Number of Bound points before refinement = {np.unique(boundary_coordinates.reshape(-1,3)).shape[0] * 0.5 + 1}")
        
        # now Get the physical tag of the boundary edges
        boundary_tags = self.mesh.cell_data["medit:ref"][0]

        # Generate a Dictionary of boundary tags and boundary coordinates
        # Keys will be the boundary tags and values will be the list of coordinates
        boundary_dict = {}
        
        # unique tags
        unique_tag = set(self.mesh.cell_data["medit:ref"][0])

        # refine the boundary points based on the number of boundary points needed
        for i in range(boundary_coordinates.shape[0]):
            p1 = boundary_coordinates[i,0,:]
            p2 = boundary_coordinates[i,1,:]
            
            if bd_sampling_method == "uniform":
                # take the current point and next point and then perform a uniform sampling 
                new_points = np.linspace(p1,p2,pow(2, boundary_point_refinement_level) + 1)
            elif bd_sampling_method == "lhs":
                # take the current point and next point and then perform a uniform sampling 
                new_points = lhs(2, pow(2, boundary_point_refinement_level) + 1)
                new_points[:,0] = new_points[:,0]*(p2[0] - p1[0]) + p1[0]
                new_points[:,1] = new_points[:,1]*(p2[1] - p1[1]) + p1[1]
            else:
                print(f'Invalid sampling method {bd_sampling_method} in {self.__class__.__name__} from {__name__}.')
                raise ValueError('Sampling method should be either uniform or lhs.')
            
            # get the boundary tag
            tag = boundary_tags[i]
            
            if tag not in boundary_dict:
                boundary_dict[tag] = new_points
            else:
                current_val = new_points
                prev_val = boundary_dict[tag]
                final = np.vstack([prev_val,current_val])
                boundary_dict[tag] = final
        
        # get unique
        for tag,bd_pts in boundary_dict.items():
            val = boundary_dict[tag]
            val = np.unique(val,axis=0)
            boundary_dict[tag] = val
            
        
        self.bd_dict = boundary_dict
        # print the new boundary points  on each boundary tag (key) in a tabular format

        total_bound_points = 0
        print(f"| {'Boundary ID':<12} | {'Number of Points':<16} |")
        print(f"| {'-'*12:<12}---{'-'*16:<16} |")
        for k, v in self.bd_dict.items():
            print(f"| {k:<12} | {v.shape[0]:<16} |")
            total_bound_points += v.shape[0]

        print(f"[INFO] : No of bound pts after refinement:  {total_bound_points}")
        

        # Assign to class values
        self.cell_points = cell_points

        # generate testvtk
        self.generate_vtk_for_test()
        
        return cell_points, self.bd_dict 


    def generate_quad_mesh_internal(self, x_limits: tuple, y_limits: tuple, n_cells_x: int, n_cells_y: int, num_boundary_points: int):
        """
        Generate and save a quadrilateral mesh with physical curves.

        Parameters:
        x_limits (tuple): The lower and upper limits in the x-direction (x_min, x_max).
        y_limits (tuple): The lower and upper limits in the y-direction (y_min, y_max).
        n_cells_x (int): The number of cells in the x-direction.
        n_cells_y (int): The number of cells in the y-direction.

        Returns:
        cell_points (numpy.ndarray): The cell points.
        bd_dict (dict): The dictionary of boundary points.

        """

        self.n_cells_x = n_cells_x
        self.n_cells_y = n_cells_y
        self.x_limits = x_limits
        self.y_limits = y_limits

        # generate linspace of points in x and y direction
        x = np.linspace(x_limits[0], x_limits[1], n_cells_x + 1)
        y = np.linspace(y_limits[0], y_limits[1], n_cells_y + 1)

        # Generate quad cells from the points
        # the output should be a list of 4 points for each cell , each being a list of 2 points [x,y]
        cells = []

        for i in range(n_cells_x):
            for j in range(n_cells_y):
                # get the four points of the cell
                p1 = [x[i], y[j]]
                p2 = [x[i+1], y[j]]
                p3 = [x[i+1], y[j+1]]
                p4 = [x[i], y[j+1]]

                # append the points to the cells
                cells.append([p1, p2, p3, p4])
        
        # convert to numpy array
        cells = np.array(cells, dtype=np.float64)

        # use arctan2 to sort the points in anticlockwise direction
        # loop over all cells and rearrange the points in anticlockwise direction
        for i in range(cells.shape[0]):
            cell = cells[i]
            # get the centroid of the cell
            centroid = np.mean(cell,axis=0)
            # get the angle of each point with respect to the centroid
            angles = np.arctan2(cell[:,1] - centroid[1], cell[:,0] - centroid[0])
            # sort the points based on the angles
            cells[i] = cell[np.argsort(angles)]

        # generate a meshio mesh object using the cells
        self.mesh = meshio.Mesh(points=cells.reshape(-1,2), cells=[("quad", cells.reshape(-1,4))])

        
        # lets generate the boundary points, this function will return a dictionary of boundary points
        # the keys will be the boundary tags and values will be the list of boundary points
        bd_points = {}

        num_bound_per_side = int(num_boundary_points/4)


        def _temp_bd_func(start, end, num_pts):
            """
            This function will return the boundary points between the start and end points
            using lhs sampling
            """
            # generate the boundary points using lhs as a np.float64 array
            # bd_pts = lhs(1, num_pts).astype(np.float64)
            bd_pts = np.linspace(0,1,num_pts).astype(np.float64)
            # scale the points
            bd_pts = bd_pts*(end - start) + start


            return bd_pts.reshape(-1)
        
        # bottom boundary
        y_bottom = (np.ones(num_bound_per_side, dtype=np.float64)*y_limits[0]).reshape(-1)
        x_bottom = _temp_bd_func(x_limits[0], x_limits[1], num_bound_per_side)
        bd_points[1000] = np.vstack([x_bottom,y_bottom]).T


        # right boundary
        x_right =(np.ones(num_bound_per_side, dtype=np.float64)*x_limits[1]).reshape(-1)
        y_right = _temp_bd_func(y_limits[0], y_limits[1], num_bound_per_side)
        bd_points[1001] = np.vstack([x_right,y_right]).T

        # top boundary
        y_top = (np.ones(num_bound_per_side, dtype=np.float64)*y_limits[1]).reshape(-1)
        x_top = _temp_bd_func(x_limits[0], x_limits[1], num_bound_per_side)
        bd_points[1002] = np.vstack([x_top,y_top]).T

        # left boundary
        x_left = (np.ones(num_bound_per_side, dtype=np.float64)*x_limits[0]).reshape(-1)
        y_left = _temp_bd_func(y_limits[0], y_limits[1], num_bound_per_side)
        bd_points[1003] = np.vstack([x_left,y_left]).T

        self.cell_points = cells
        self.bd_dict = bd_points

        # generate vtk 
        self.generate_vtk_for_test()

        return self.cell_points , self.bd_dict
    
    def generate_vtk_for_test(self):
        """
        generate a VTK from Mesh file (External) or using gmsh (for Internal)

        Parameters:
        None
        """

        if(self.mesh_generation_method == "internal"):
            # initialise the mesh
            gmsh.initialize()

            # Now, lets generate the mesh with the points. 
            x_range = self.x_limits[1] - self.x_limits[0] 
            y_range = self.y_limits[1] - self.y_limits[0] 

            mesh_size_x = x_range/self.n_test_points_x
            mesh_size_y = y_range/self.n_test_points_y


            # generate a gmsh with the given parameters
            Xmin = self.x_limits[0]
            Xmax = self.x_limits[1]
            Ymin = self.y_limits[0]
            Ymax = self.y_limits[1]

            point1 = gmsh.model.geo.add_point(Xmin, Ymin, 0, mesh_size_x)
            point2 = gmsh.model.geo.add_point(Xmax, Ymin, 0, mesh_size_x)
            point3 = gmsh.model.geo.add_point(Xmax, Ymax, 0, mesh_size_y)
            point4 = gmsh.model.geo.add_point(Xmin, Ymax, 0, mesh_size_y)

            line1 = gmsh.model.geo.add_line(point1, point2,1000) ## Bottom
            line2 = gmsh.model.geo.add_line(point2, point3,1001) ## Right
            line3 = gmsh.model.geo.add_line(point3, point4,1002) ## Top
            line4 = gmsh.model.geo.add_line(point4, point1,1003) ## Left           

            face1 = gmsh.model.geo.add_curve_loop([line1, line2, line3, line4])

            gmsh.model.geo.add_plane_surface([face1])

            # Create the relevant Gmsh data structures
            # from Gmsh model.
            gmsh.model.geo.synchronize()
            
            # Generate mesh:
            gmsh.model.mesh.generate()  

            # ## Add Physical Groups for ParMooN-Import
            # ## Bottom - 1000, Right 1001, Top 1002 , Left 1003
            # ## (Dimension, Line/Curve/Surface , Physical Group Number)
            # gmsh.model.addPhysicalGroup(1, [line1], 1000)
            # gmsh.model.addPhysicalGroup(1, [line2], 1001)
            # gmsh.model.addPhysicalGroup(1, [line3], 1002)
            # gmsh.model.addPhysicalGroup(1, [line4], 1003)

            # # Generate mesh:
            # gmsh.model.mesh.generate()

            # ## save all Elements
            # gmsh.option.setNumber("Mesh.SaveAll", 1)

            mesh_file_name = Path(self.output_folder) / "internal.msh"
            vtk_file_name  = Path(self.output_folder) / "internal.vtk"

            gmsh.write(str(mesh_file_name))
            print("[INFO] : Internal mesh file generated at ", str(mesh_file_name))

            # read the mesh using meshio
            mesh = meshio.gmsh.read(str(mesh_file_name))
            meshio.vtk.write(str(vtk_file_name), mesh, binary=False, fmt_version="4.2")

            print("[INFO] : VTK file for internal mesh file generated at ", str(mesh_file_name))

        elif(self.mesh_generation_method == "external"):

            vtk_file_name  = Path(self.output_folder) / "external.vtk"

            # Use the internal mesh to generate the vtk file 
            mesh = meshio.read(str(self.mesh_file_name))
            meshio.vtk.write(str(vtk_file_name), mesh, binary=False, fmt_version="4.2")

            print("[INFO] : VTK file for external mesh file generated at ", str(vtk_file_name))



        else:
            # print the file name and function name 
            print("[Error] : File : geometry_2d.py, Function: ")
            raise Exception("Unknown mesh type")
    
    def get_test_points(self):
        """
        This function is used to extract the test points from the given mesh

        Parameters:
        None

        Returns:
        test_points (numpy.ndarray): The test points for the given domain
        """

        if(self.mesh_generation_method == "internal" or self.mesh_generation_method == "convolution"):
            # vtk_file_name  = Path(self.output_folder) / "internal.vtk"
            # code over written to plot from np.linspace instead of vtk file
            # generate linspace of points in x and y direction based on x and y limits
            x = np.linspace(self.x_limits[0], self.x_limits[1], self.n_test_points_x)
            y = np.linspace(self.y_limits[0], self.y_limits[1], self.n_test_points_y)
            # generate meshgrid
            X, Y = np.meshgrid(x, y)
            # stack the points
            self.test_points = np.vstack([X.flatten(), Y.flatten()]).T

            return self.test_points


        elif(self.mesh_generation_method == "external"):
            vtk_file_name  = Path(self.output_folder) / "external.vtk"
        else:
            # print the file name and function name 
            print("[Error] : File : geometry_2d.py, Function: ")
            raise Exception("Unknown mesh type")


        mesh = meshio.read(str(vtk_file_name))
        points = mesh.points
        return points[:,0:2]    # return only first two columns
    

    def write_vtk(self, solution, output_path, filename, data_names):
        """
        This function will write the data to vtk file

        Parameters:
        - solution (numpy.ndarray): The solution vector
        - output_path (str): The path to the output folder
        - filename (str): The name of the output file
        - data_names (list): The list of data names in vtk file to be written as scalars

        Returns:
        None
        """
        # read the existing vtk into file 
        if(self.mesh_generation_method == "internal"):
            vtk_file_name  = Path(self.output_folder) / "internal.vtk"
        elif(self.mesh_generation_method == "external"):
            vtk_file_name  = Path(self.output_folder) / "external.vtk"
        
        data = []
        with open(vtk_file_name,'r') as File:
            for line in File:
                data.append(line)

        # get the output file name
        output_file_name = Path(output_path) / filename

        if(solution.shape[1] != len(data_names)):
            print("[Error] : File : geometry_2d.py, Function: write_vtk")
            print("Num Columns in solution = ", solution.shape[1] , " Num of data names = ", len(data_names))
            raise Exception("Number of data names and solution columns are not equal")
        

        # write the data to the output file
        with open(str(output_file_name),'w') as FN:
            for line in data:
                FN.write(line)
                if 'POINT_DATA' in line.strip():
                    break
                pass

            for i in range(solution.shape[1]):
                FN.write('SCALARS ' + data_names[i] + ' float\n')
                FN.write('LOOKUP_TABLE default\n')
                np.savetxt(FN, solution[:,i])
                FN.write('\n')

        # save the vtk file as image
        self.save_vtk_as_image(str(output_file_name), data_names)

    def save_vtk_as_image(self, vtk_file_name, scalars_list):
        """
        Function to save VTK as image for easy rendering. Note : only works on linux environment
        # If any problem exists, comment the line pv.start_xvfb() in the constructor and also
        comment this function

        Parameters:
        - vtk_file_name (str): The name of the vtk file
        - scalars_list (list): The list of scalars to be plotted

        Returns:
        None
        """
        import gc  # Import the garbage collector

        # Read the VTK file
        vtk_data = pv.read(vtk_file_name)

        # Setup the plotter
        for scal in scalars_list:
            if scal == "Exact":
                continue
            plotter = pv.Plotter(off_screen=True, window_size=[640, 480], image_scale=2)
            # Set the background to white
            plotter.set_background('white')

            plotter.add_mesh(vtk_data, scalars=scal, cmap="jet", show_scalar_bar=False)

            plotter.view_xy()

            if scal == "Sol":
                # using the Path object, remove the extension and add png
                output_image_path = Path(vtk_file_name).with_suffix('.png')
                # Adding a vertical scalar bar
                plotter.add_scalar_bar(vertical=True, title='Solution', title_font_size=20, label_font_size=20, width=0.25, position_x=0.80, position_y=0.25)
            else:
                # get the file name from the path object
                file_name = Path(vtk_file_name).name
                # remove the extension and obtain the epoch number
                # file name will be of the form prediction_4000.vtk
                epoch = file_name.split('.')[0].split('_')[1]

                # add the epoch number to the output file name
                output_image_path = Path(self.output_folder) / (scal + "_" + epoch + ".png")

                # Adding a vertical scalar bar
                plotter.add_scalar_bar(vertical=True, title=scal, title_font_size=20, label_font_size=20, width=0.25, position_x=0.80, position_y=0.25)

            # Save the screenshot
            plotter.screenshot(output_image_path)

            # Close the plotter
            plotter.close()

            # Delete the plotter and vtk_data objects
            del plotter
            

            # Call the garbage collector
            gc.collect()
        del vtk_data

    ## FBVPINN additions by Divij. 10.1.2024
    def generate_conv_domains(self, x_limits, y_limits, n_cells_x: int, n_cells_y: int, kernel_size_row: int, kernel_size_col: int, stride_row: int, stride_col: int):
        """
        Generate cells and blocks using a convolutional approach.

        Args:
            x_limits: Tuple containing the lower and upper limits in the x-direction.
            y_limits: Tuple containing the lower and upper limits in the y-direction.
            n_cells_x: Number of cells in the x-direction.
            n_cells_y: Number of cells in the y-direction.
            kernel_size_row: Number of cells in the row direction of the kernel.
            kernel_size_col: Number of cells in the column direction of the kernel.
            stride_row: Number of cells to skip in the row direction.
            stride_col: Number of cells to skip in the column direction.
        

        Returns:
            blocks: List of blocks.
            grid_x: List of x-coordinates of the grid.
            grid_y: List of y-coordinates of the grid.
        """
        self.n_cells_x = n_cells_x
        self.n_cells_y = n_cells_y
        n_cells = self.n_cells_x * self.n_cells_y
        self.cells_in_domain = np.arange(n_cells)
        self.cells_in_domain_2d = np.arange(n_cells).reshape((n_cells_x, n_cells_y))
        self.cells_in_domain_2d = np.flip(self.cells_in_domain_2d, axis=0)

        self.kernel_size_row = kernel_size_row
        self.kernel_size_col = kernel_size_col
        self.stride_row = stride_row
        self.stride_col = stride_col

        self.domains = []
        rows, cols = self.cells_in_domain_2d.shape
        for i in range(0, rows-kernel_size_row+1, stride_row):
            for j in range(0, cols-kernel_size_col+1, stride_col):
                domain = []
                for k in range(i, i+kernel_size_row):
                    for l in range(j, j+kernel_size_col):
                        domain.append(self.cells_in_domain_2d[k][l])
                self.domains.append(domain)
        # Generate the grid
        self.x_left = x_limits[0]
        self.x_right = x_limits[1]
        self.y_bottom = y_limits[0]
        self.y_top = y_limits[1]

        delta_x = (self.x_right - self.x_left)/n_cells_x
        delta_y = (self.y_top - self.y_bottom)/n_cells_y

        self.grid_x = np.asarray([self.x_left + i*delta_x for i in range(n_cells_x+1)])
        self.grid_y = np.asarray([self.y_bottom + i*delta_y for i in range(n_cells_y+1)])

        
        # calculate span and mean of each subdomain
        self.calculate_subdomain_means()
        self.calculate_subdomain_spans()
        
        self.print_domain_info()

        
        
        return self.cells_in_domain, self.domains, self.grid_x, self.grid_y
    
    def generate_quad_mesh_convolutional(self, x_limits: tuple, y_limits: tuple, n_cells_x: int, n_cells_y: int, num_boundary_points: int):
        """
        Generate and save a quadrilateral mesh with physical curves.

        Parameters:
        x_limits (tuple): The lower and upper limits in the x-direction (x_min, x_max).
        y_limits (tuple): The lower and upper limits in the y-direction (y_min, y_max).
        n_cells_x (int): The number of cells in the x-direction.
        n_cells_y (int): The number of cells in the y-direction.
        """

        self.n_cells_x = n_cells_x
        self.n_cells_y = n_cells_y
        self.x_limits = x_limits
        self.y_limits = y_limits

        # generate linspace of points in x and y direction
        x = np.linspace(x_limits[0], x_limits[1], n_cells_x + 1)
        y = np.linspace(y_limits[0], y_limits[1], n_cells_y + 1)

        # Generate quad cells from the points
        # the output should be a list of 4 points for each cell , each being a list of 2 points [x,y]
        cells = []

        for i in range(n_cells_y):
            for j in range(n_cells_x):
                # get the four points of the cell
                p1 = [x[j], y[i]]
                p2 = [x[j+1], y[i]]
                p3 = [x[j+1], y[i+1]]
                p4 = [x[j], y[i+1]]

                # append the points to the cells
                cells.append([p1, p2, p3, p4])
        
        # convert to numpy array
        # cells = np.array(cells, dtype=np.float64)
                
         # lets generate the boundary points, this function will return a dictionary of boundary points
        # the keys will be the boundary tags and values will be the list of boundary points
        

        

        boundary_limits = []
        for block_id, cells_in_current_block in enumerate(self.domains):
            x_min = self.x_right
            x_max = self.x_left
            y_min = self.y_top
            y_max = self.y_bottom
            for cell in cells_in_current_block:
                ex = cell%self.n_cells_x
                ey = cell//self.n_cells_x
                x_min = min(x_min, self.grid_x[ex])
                x_max = max(x_max, self.grid_x[ex+1])
                y_min = min(y_min, self.grid_y[ey])
                y_max = max(y_max, self.grid_y[ey+1])

            boundary_limits.append([x_min, x_max, y_min, y_max])

        return cells, boundary_limits
    
    def assign_sub_domain_coords(self, cells_in_subdomain, cells_points):
        """
        Assigns coordinates to the subdomains.

        Args:
            cells_in_subdomain: List of cells in the subdomain.
            cells_points: List of points in the cells.
        
        Returns:
            cells_points_subdomain: List of points in the subdomains.
        """
        cells_points_subdomain = []
        for cell in cells_in_subdomain:
            cells_points_subdomain.append(cells_points[cell])
        return np.array(cells_points_subdomain, dtype=np.float64)


    def calculate_subdomain_means(self):
        self.x_mean_list = []
        self.y_mean_list = []
        self.xmin_list = []
        self.xmax_list = []
        self.ymin_list = []
        self.ymax_list = []
        for block_id, cells_in_current_block in enumerate(self.domains):
            x_min = self.x_right
            x_max = self.x_left
            y_min = self.y_top
            y_max = self.y_bottom
            for cells in cells_in_current_block:
                ex = cells%self.n_cells_x
                ey = cells//self.n_cells_x
                x_min = min(x_min, self.grid_x[ex])
                x_max = max(x_max, self.grid_x[ex+1])
                y_min = min(y_min, self.grid_y[ey])
                y_max = max(y_max, self.grid_y[ey+1])

            x_mean = np.float64((x_min + x_max)/2.0)
            y_mean = np.float64((y_min + y_max)/2.0)
            self.x_mean_list.append(x_mean)
            self.y_mean_list.append(y_mean)
            self.xmin_list.append(x_min)
            self.xmax_list.append(x_max)
            self.ymin_list.append(y_min)
            self.ymax_list.append(y_max)

        # print(f"X mean list = {self.x_mean_list}")
        # print(f"Y mean list = {self.y_mean_list}")

        return self.x_mean_list, self.y_mean_list 

    def calculate_subdomain_spans(self):
        self.x_span_list = []
        self.y_span_list = []
        for block_id, cells_in_current_block in enumerate(self.domains):
            x_min = self.x_right
            x_max = self.x_left
            y_min = self.y_top
            y_max = self.y_bottom
            for cells in cells_in_current_block:
                ex = cells%self.n_cells_x
                ey = cells//self.n_cells_x
                x_min = min(x_min, self.grid_x[ex])
                x_max = max(x_max, self.grid_x[ex+1])
                y_min = min(y_min, self.grid_y[ey])
                y_max = max(y_max, self.grid_y[ey+1])

            x_span = np.float64(x_max - x_min)
            y_span = np.float64(y_max - y_min)
            self.x_span_list.append(x_span)
            self.y_span_list.append(y_span)

        # print(f"X span list = {self.x_span_list}")
        # print(f"Y span list = {self.y_span_list}")

        return self.x_span_list, self.y_span_list

    def print_domain_info(self):
        print(f"Number of subdomains = {len(self.domains)}")
        
        title = "Subdomain Information"
        cols = ["Subdomain ID", "Cell IDs"]
        # print_table(title, cols, list(range(len(self.domains))), self.domains)

        title = "Subdomain Params"
        cols = ["Subdomain ID", "x_min", "x_max", "y_min", "y_max", "x_mean", "y_mean", "x_span", "y_span"]
        # print_table_multicolumns(title, cols, 
                                #  [list(range(len(self.domains))), self.xmin_list, self.xmax_list, self.ymin_list, self.ymax_list, self.x_mean_list, self.y_mean_list, self.x_span_list, self.y_span_list])
            
        
        return None
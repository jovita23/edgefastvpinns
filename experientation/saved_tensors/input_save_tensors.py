# This YAML file contains configuration parameters for Edge-FastVPINNs cd2d gear problem
# Run to save the tensors for different decompositions and different ranks

experimentation:
  output_path: "output/test/"
  seed: 1234
  

decomposition_type: "ttd"  # "tucker" or "cp" or "ttd"

geometry:
  mesh_generation_method: "external"
  generate_mesh_plot: True
  internal_mesh_params:
    x_min: 0
    x_max: 1
    y_min: 0
    y_max: 1
    n_cells_x: 8
    n_cells_y: 8
    n_boundary_points: 2000
    n_test_points_x: 100
    n_test_points_y: 100
  
  exact_solution:
    exact_solution_generation: "external" # whether the exact solution needs to be read from external file.
    exact_solution_file_name: "fem_output_gear_forward_sin.csv" # External solution file name.

  mesh_type: "quadrilateral"
  external_mesh_params:
    mesh_file_name: "meshes/gear.mesh"  # should be a .mesh file
    boundary_refinement_level: 2
    boundary_sampling_method: "uniform"  # "uniform" 

fe:
  fe_order: 4 
  fe_type: "jacobi"   
  quad_order: 5
  quad_type: "gauss-jacobi"  
  rank_list : [25,16,25]  #change just the first item on this list. For cp and ttd, the rank will be rank_list[0]


pde:
  beta: 5
model:
  model_architecture: [2, 50,50,50, 1]
  activation: "tanh"
  use_attention: False
  epochs: 100000
  dtype: "float32"
  set_memory_growth: False
  learning_rate:
    initial_learning_rate: 0.005
    use_lr_scheduler: True
    decay_steps: 1000
    decay_rate: 0.99
    staircase: False

logging:
  update_progress_bar: 1000 # Number of steps between each update of the progress bar.
  update_console_output: 1000  # Number of steps between each update of the console output.
  update_solution_images: 1000  # Number of steps between each update of the intermediate solution images.
  plot_residual_images: True
  print_verbose: True  # Flag indicating whether to print verbose output.
  test_errors_last_n_epochs: 1000


wandb:
  use_wandb: False
  project_name: "gear best Tucker Decomposition"
  wandb_run_prefix: "gear_10"
  entity: "starslab-iisc"

additional:
  run_by: "Thivin"  # Name of the person running the experiment.
  System: "24"  # System identifier. 


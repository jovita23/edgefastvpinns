
# Inverse CD2D






## 1. Understanding File Structures

To run CD2D problem on gear mesh using the decomposed tensor to calculate the loss, we need to modify the parameters in the following files :


#### Main file (`main_inverse.py`)
    
    
This file is responsible for solving the inverse cd2d gear problem. It takes modules and functions from additional files and runs the inverse problem. 





#### Model File (Eg: `src/model/model_inverse_domain_decomposed.py`)

This file is responsible for creating the model. The model is created using the `Model` class.  Any NN architecture can be implemented within this model class as inputs and output dimensions are handled accordingly.




#### Physics File (Eg: `src/physics/cd2d_tucker_inv.py`)

This file is responsible for creating the loss function for the inverse problem using the decomposed tensor. Any decomposition on tensor can be implemented within this physics class such as tucker (`src/physics/cd2d_tucker_inv.py`) or ttd - Tensor Train decomposition (`src/physics/cd2d_ttd_inv.py`) or cp - Canonical polyadic (`src/physics/cd2d_cp_inv.py`).



#### Example file (Eg: `examples/inverse_gear.py`)

This file stores the information for a specific training example. It contains the following information such as 
- `exact_solution`: The exact solution of the problem
- `boundary_condition`: The boundary condition of the problem
- `boundary_values`: The boundary values of the problem
- `bilinear_parameters` : Bilinear parameters of PDE such as epsilon, mu, etc.


#### Decomposition file (Eg: `src/data/tucker_decom.py`)

This file contains the decomposition function which decomposes the tensor using the desired decomposition type. The tensor can be decomposed using any decomposition by using these files: tucker (`src/data/tucker_decom.py`) or ttd - Tensor Train decomposition (`src/data/ttd_decom.py`) or cp - Canonical polyadic (`src/data/cp_decom.py`).

The corresponding example file can be imported into the `main_edge_decomposition.py` file using the following line:


```python
# import the example file
from src.data.tucker_decom import *
```

### Update input_edge_decomposition.yaml file. 

The `input_inverse_domain_decom.yaml` file has lot of configuration parameters related to the problem such as the fe_order, quad_order, the number of boundary points, number of test points, learning rate, NN Architecture, output_folder etc. Change the decomposition type: 'tucker' or 'cp' or 'ttd' in input_edge_decomposition.yaml file according to the desired decomposition method.



## 2. Running the Code

To run the code, from the root folder of the repository, run the following command:

```bash
python3 main_inverse.py input_inverse_domain_decom.yaml
```



# Main File for running the Python code
# of all the cells within the given mesh
# Author: Thivin Anandh D
# Date:  02/Nov/2023


import numpy as np
import os
def save_contour_data(x, y, z, output_path, filename, title=None):
    """
    This function saves x, y, z arrays into a binary file.
    """
    np.savez(os.path.join(output_path, filename), x=x, y=y, z=z)

def save_line_data(x, y, output_path, filename, title=None):
    """
    This function saves x, y arrays into a binary file.
    """
    np.savez(os.path.join(output_path, filename), x=x, y=y)
    
def load_contour_data(file_path):
    """
    This function loads x, y, z arrays from a binary file.
    """
    data = np.load(file_path)
    x = data['x']
    y = data['y']
    z = data['z']
    return x, y, z

def save_loss_function(loss_function, output_path, filename):
    """
    This function saves the loss function array into a binary file.
    """
    np.save(os.path.join(output_path, filename), loss_function)

def load_loss_function(file_path):
    """
    This function loads the loss function array from a binary file.
    """
    loss_function = np.load(file_path)
    return loss_function


def save_array(array, output_path, filename):
    """
    This function saves the array into a binary file.
    """
    np.save(os.path.join(output_path, filename), array)

def load_array(file_path):
    """
    This function loads the array from a binary file.
    """
    array = np.load(file_path)
    return array


def save_test_loss_function(loss_function, output_path, fileprefix=""):
    """
    This function saves the loss function array into a binary file.
    """
    if fileprefix == "":
        filename = "test_loss_function.npy"
    else:
        filename = fileprefix + "_test_loss_function.npy"
    np.save(os.path.join(output_path, filename), loss_function)

def load_test_loss_function(file_path):
    """
    This function loads the loss function array from a binary file.
    """
    loss_function = np.load(file_path)
    return loss_function

def print_table_edge(title, column_headers, column1, column2):
    # Print table title
    print(title)
    print(title)
    print("=" * len(title))  # Print underline of '=' characters


    # Print column headers
    num_columns = len(column_headers)
    print("-" * (30 * num_columns))  # Print underline of '=' characters

    for header in column_headers:
        print(f"{header:<30} | ", end="")  # Left-align headers within 40 characters
    print()  # Move to the next line after printing headers

    print("-" * (30 * num_columns))  # Print underline of '=' characters

    # Print values row by row
    for c1,c2 in zip(column1, column2):
        print(f"{c1:<40}: {c2}")
    print()  # Move to the next line after printing all rows
    print("-" * (30 * num_columns))  # Print underline of '=' characters
    print()
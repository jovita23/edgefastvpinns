#function for decomposing the tensor



import tensorflow as tf
import tensorly as tl
from tensorly.decomposition import tucker
import numpy as np


def decomposition(tensor,ranks):
    core,factors = tucker(tensor,rank = ranks)
    return core,factors
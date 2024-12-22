#function for decomposing the tensor

import tensorflow as tf
import tensorly as tl
from tensorly.decomposition import parafac
import numpy as np


def decomposition(tensor,ranks):
    core,factors = parafac(tensor.numpy(),rank = ranks[0])
    return core,factors
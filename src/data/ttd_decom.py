#function for decomposing the tensor


import tensorflow as tf
import t3f
import numpy as np


def decomposition(tensor,ranks):
    T3_tensor  = t3f.to_tt_tensor(tensor, max_tt_rank=ranks[0])
    core = 1.0
    return core, T3_tensor
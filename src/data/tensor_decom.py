#function for decomposing the tensor



import tensorflow as tf
import tensorly as tl

import t3f
from tensorly.decomposition import tucker
import numpy as np
from tensorly.decomposition import parafac


def decomposition(tensor,ranks,type):

    if type == 'tucker':
        core,factors = tucker(tensor.numpy(),rank = ranks)

    elif type == 'cp':
        core,factors = parafac(tensor.numpy(),rank = ranks[0])
    elif type == 'ttd': 
        factors = t3f.to_tt_tensor(tensor, max_tt_rank=ranks[0])
        core = 1.0

    
    return core,factors
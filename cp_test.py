import tensorflow as tf
import tensorly as tl
from tensorly.decomposition import tucker
import numpy as np
from tensorly.decomposition import parafac


n_cells = 16
n_test = 25
n_quad = 20


original_tensor = np.random.rand(n_cells, n_test, n_quad)
original_tensor = tf.convert_to_tensor(original_tensor)

original_matrix = np.random.rand(n_cells, n_quad)
original_matrix = tf.convert_to_tensor(original_matrix)

original_residiual  = tf.linalg.matvec(original_tensor, original_matrix)


from tensorly.decomposition import parafac

def cp_decomposition(tensor,rank):

    weights, factors = parafac(tensor.numpy(), rank=rank)

    

    c = tf.einsum('kr, ik -> ir',factors[2],original_matrix)
    c = tf.einsum('ir, jr -> ijr',c,factors[1])
    c = tf.einsum('ijr, ir -> ij',c,factors[0])

    return c


def cp_reconstruction(tensor,rank):

    weights, factors = parafac(tensor.numpy(), rank=rank)

    cp_org = tl.cp_to_tensor((weights,factors))

   
    c = tf.einsum('ir, jr -> ijr',factors[0],factors[1])
    c = tf.einsum('ijr, kr -> ijk',c,factors[2])

    return cp_org,c






def tucker_decomposition(tensor, rank_list):

    core, factors = tucker(original_tensor, rank = rank_list)

    c= tf.einsum('mnp, kp -> mnk', core, factors[2])
    c = tf.einsum('mnk, ik -> mni', c, original_matrix)

   
    c = tf.einsum('mni, jn -> imj', c, factors[1])
    c = tf.einsum('imj, im -> ij', c, factors[0])

    return c


print(original_residiual)


def compare_tensors(tensom, tenson, tolerance=1e-4):
    norms = {
        "norm1" : (tf.norm(tensom, ord=1),tf.norm(tenson, ord=1)  ),
        "norm2" : (tf.norm(tensom, ord=2),tf.norm(tenson, ord=2)  ),
        # "Frob" : (tf.norm(tensom, ord="fro"),tf.norm(tenson, ord="fro")  ),
        "inf" : (tf.norm(tensom, ord=np.inf),tf.norm(tenson, ord=np.inf)  ),
        "min" : (np.min(tensom), np.min(tenson)),
        "max" : (np.max(tensom), np.max(tenson)),
        "mean" : (np.mean(tensom), np.mean(tenson))
    }
    
    print("Shape of Tensor 1 : " , tensom.shape)
    print("Shape of Tensor 2 : " , tenson.shape)


    header = f"{'Value':<10} {'Tensor-1':<12} {'Tensor-2':<12} {'Difference':<12} {'Result':<12}"
    print(header)
    print("=" * len(header))
    
    value_list = []

    for norm_name, (norm1, norm2) in norms.items():
        diff = abs(norm1 - norm2)
        if(diff < tolerance):
            value = "Pass"
        else:
            value = "Fail"
        print(f"{norm_name:<10} {norm1:.3e}   {norm2:.3e}   {diff:.3e} {value}")
        value_list.append(value)
    
    count = sum([1 for k in value_list if k == "Pass"])

    print("All Close :  " , np.allclose(tensom, tenson))

cp_tensor = cp_decomposition(original_tensor,rank = 600)
tucker_tensor = tucker_decomposition(original_tensor,[16,25,20])
#cp_org_tensor,cp_recon_tensor = cp_reconstruction(original_tensor,rank = 600)
compare_tensors(original_residiual,cp_tensor)
compare_tensors(original_residiual,tucker_tensor)
# compare_tensors(original_tensor,original_tensor)
#compare_tensors(cp_org_tensor,cp_recon_tensor)

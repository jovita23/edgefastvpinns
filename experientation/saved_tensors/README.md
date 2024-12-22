# Save tensors to disk 

The decomposed tensors will be saved to disk in a numpy array and then they will be used for computation in the Edge devices

## To Run :
Copy all these files to the main directory of the project to run them.


## Notes 

- The fespace object is generated once and saved as `fespace.pkl` in the directory. This is to avoid recomputation of the fespace object for every scenario. 

- For CP only the following ranks were run 
```
tucker_factors = [2,4,6,8,10,12,14]
decomposition_types = ["cp"]
```

- For Tucker and TT the following ranks were run 
```
tucker_factors = [2,4,16,64,256]
decomposition_types = ["tucker", "tt"]
```

- For SVD, the fespace itself will be constructed using the SVD decomposition and it can be done within the edge device itself.
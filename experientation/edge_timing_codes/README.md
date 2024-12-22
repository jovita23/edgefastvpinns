# Edge Timing Codes

These are codes, that will read the saved tensor decompositions (generated from a larger system with GPU) and will be used to time the training time of various decompositions and various ranks within the Edge Environment.


## Run :

Copy the files to the main directory of the project to run them.

## Notes:

- The `fespace.pkl` and the `datahandler.pkl` files are available to be read directly from the disk. 
- The tensor decompositions are read from the disk and the training time is recorded for each decomposition and rank.

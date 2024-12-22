# Scheduler to save the tensors for different decompositions and different ranks

import numpy as np
import pandas as pd
import os
import sys
import subprocess
import yaml
import pathlib
from pathlib import Path

if __name__ == "__main__":

    # throw error if the input_script.yaml file is not provided
    if len(sys.argv) != 2:
        print("[ERROR] : Please provide the input_script.yaml file")
        exit(1)

    # read the input_script.yaml file
    with open(sys.argv[1], 'r') as f:
        config = yaml.safe_load(f)

    experiment_name = "Gear_experiment_TTD_seeding"
    
    # tucker_factors = [2,4,16,64,256]
    tucker_factors = [2,4,6,8,10,12,14]
    seeds = [1234]
    # decomposition_types = ["ttd", "tucker", "cp"]
    decomposition_types = ["cp"]
    
    for rank in  tucker_factors :
        for seed in seeds:
            for decomposition_type in decomposition_types:
                print("====================================================================================================")
                print("====================================================================================================")
                print(" RUNNING EXPERIMENT FOR rank = {}, seed = {}, decomposition_type = {}".format(rank, seed, decomposition_type))
                print("====================================================================================================")
                print("====================================================================================================")
                
                config["experimentation"]["output_path"] = f"output/saved_tensors/{decomposition_type}"
                config["experimentation"]["seed"] = seed
                config["decomposition_type"] = decomposition_type
                config["fe"]["rank_list"] = [rank,16,25]

                # Experiment name 
                config["wandb"]["project_name"] = experiment_name
                config["wandb"]["wandb_run_prefix"] = f"gear_{rank}_seed_{seed}_{decomposition_type}"

                output_file = f"gear_rank_{rank}_seed_{seed}_{decomposition_type}.txt"

                with open("temp.yaml", 'w') as f:
                    yaml.dump(config, f)

                # Run the experiment, Span a subprocess and wait for that subprocess to complete
                # Create a file name under experiment_logs_folder, with name as the experiment name and the run prefix
                with open(output_file, 'w') as f:
                    completed_process = subprocess.run(["python3", "main_save_tensors.py", "temp.yaml"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                    for line in completed_process.stdout.decode().splitlines():
                        print(line)
                        f.write(line + "\n")

                # Check if the subprocess completed successfully
                if completed_process.returncode != 0:
                    print("[ERROR] : Experiment failed for rank = {}".format(rank))
                
                # Remove the temp.yaml file
                os.remove("temp.yaml")

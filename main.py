import torch
import the_well
from torch.utils.data import DataLoader
import numpy as np
import os
import pickle
from the_well.data import WellDataset
import matplotlib
matplotlib.use('TkAgg') # or 'Qt5Agg', 'WXAgg', etc.
import dask.array as da
import matplotlib.pyplot as plt
from sklearn.utils.extmath import randomized_svd
from randomized_svd import optimal_rank, optimal_rank_elbow
import scipy.io
names_of_datasets_top_run = ["MHD_64", "turbulent_radiative_layer_2D", "post_neutron_star_merger","viscoelastic_instability",  "helmholtz_staircase"]
for dataset in names_of_datasets_top_run:
    print(f"Processing dataset: {dataset}")
    trainset = WellDataset(well_base_path="E:\Datasets\datasets", well_dataset_name=f"{dataset}", well_split_name="train")
    train_loader = DataLoader(trainset)
    results_dir = "E:"
    for field_idx in range(next(iter(train_loader))["input_fields"].shape[-1]): 
        count = 0
        per_batch_in = None
        per_batch_out = None
        for batch in train_loader: 
            count += 1 
            print(f"Processing batch {count}")
            if count == 1: 
                if len(batch["input_fields"].shape) == 6:
                    per_batch_in = np.zeros((len(train_loader.dataset), batch["input_fields"].shape[2]*batch["input_fields"].shape[3]*batch["input_fields"].shape[4]), dtype=np.uint8) 
                    per_batch_out = np.zeros((len(train_loader.dataset), batch["output_fields"].shape[2]*batch["output_fields"].shape[3]*batch["output_fields"].shape[4]), dtype=np.uint8)
                    input_field = batch["input_fields"][:, :, :, :, :, field_idx]
                    output_field = batch["output_fields"][:, :, :, :, :, field_idx]  
                else:
                    per_batch_in = np.zeros((len(train_loader.dataset), batch["input_fields"].shape[1]*batch["input_fields"].shape[2]*batch["input_fields"].shape[3]), dtype=np.uint8) 
                    per_batch_out = np.zeros((len(train_loader.dataset), batch["output_fields"].shape[1]*batch["output_fields"].shape[2]*batch["output_fields"].shape[3]), dtype=np.uint8)
                    input_field = batch["input_fields"][:, :, :, :, field_idx]
                    output_field = batch["output_fields"][:, :, :, :,field_idx]  
            inputs_data = input_field.reshape(input_field.shape[0] * input_field.shape[1], -1) # Process output field 
            output_data = output_field.reshape(output_field.shape[0] * output_field.shape[1], -1) 
            per_batch_in[count-1, :] = inputs_data 
            per_batch_out[count-1, :] = output_data # Clear intermediate variables to free memory 
            del inputs_data, output_data 
        # Create results directory if it doesn't exist
        # First compute SVD to get singular values
        U_temp, S_temp, _ = randomized_svd(per_batch_in[:, :], n_components=min(per_batch_in.shape[0], per_batch_in.shape[1], 1000), n_iter=5, random_state=None)
        m, n = per_batch_in[:, :].shape
        estimated_rank = optimal_rank(m, n, S_temp)
        print("Estimated Rank: ", estimated_rank)
        U, S, Vt = randomized_svd(per_batch_in[:, :], n_components=estimated_rank, n_iter=5, random_state=None)
        U_out, S_out, Vt_out = randomized_svd(per_batch_out[:, :], n_components=estimated_rank, n_iter=5, random_state=None)   
        scipy.io.savemat(f"output_matrix_rank_{estimated_rank}_{dataset}_{field_idx}.mat", {f"Matrix_1_{dataset}_in_U{field_idx}": U, f"Matrix_1_{dataset}_in_S{field_idx}": S, f"Matrix_1_{dataset}_in_Vt{field_idx}": Vt, f"Matrix_1_{dataset}_out_U{field_idx}" : U_out, f"Matrix_1_{dataset}_out_S{field_idx}": S_out, f"Matrix_1_{dataset}_out_Vt{field_idx}": Vt_out})
        del per_batch_in, per_batch_out, estimated_rank
    print("Field-wise global SVD complete.")
    
    
    
    

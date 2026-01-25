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
from randomized_svd import optimal_rank
import scipy.io
standby = ["euler_multi_quadrants_periodicBC", "MHD_256", "euler_multi_quadrants_openBC"]

for dataset in standby:
    print(f"Processing dataset: {dataset}")
    trainset = WellDataset(well_base_path="/media/david/USB-HDD/Datasets/datasets", well_dataset_name=f"{dataset}", well_split_name="train")
    train_loader = DataLoader(trainset)
    results_dir = "/media/david/USB-HDD/"
    for field_idx in range(4, next(iter(train_loader))["input_fields"].shape[-1]): 
        count = 0
        per_batch_in = None
        per_batch_out = None
        for batch in train_loader:
            print(batch["input_fields"].shape)
            count += 1 
            print(f"Processing batch {count}")
            if count == 1: 
                if len(batch["input_fields"].shape) == 6:
                    per_batch_in = np.zeros((len(train_loader.dataset), batch["input_fields"].shape[2]*batch["input_fields"].shape[3]*batch["input_fields"].shape[4]), dtype=np.float32) 
                    per_batch_out = np.zeros((len(train_loader.dataset), batch["output_fields"].shape[2]*batch["output_fields"].shape[3]*batch["output_fields"].shape[4]), dtype=np.float32)
                    input_field = batch["input_fields"][:, :, :, :, :, field_idx]
                    output_field = batch["output_fields"][:, :, :, :, :, field_idx]  
                else:
                    per_batch_in = np.zeros((len(train_loader.dataset), batch["input_fields"].shape[1]*batch["input_fields"].shape[2]*batch["input_fields"].shape[3]), dtype=np.float32) 
                    per_batch_out = np.zeros((len(train_loader.dataset), batch["output_fields"].shape[1]*batch["output_fields"].shape[2]*batch["output_fields"].shape[3]), dtype=np.float32)
                    input_field = batch["input_fields"][:, :, :, :, field_idx]
                    output_field = batch["output_fields"][:, :, :, :,field_idx]  
            inputs_data = input_field.reshape(input_field.shape[0] * input_field.shape[1], -1) # Process output field 
            output_data = output_field.reshape(output_field.shape[0] * output_field.shape[1], -1) 
            per_batch_in[count-1, :] = inputs_data 
            per_batch_out[count-1, :] = output_data # Clear intermediate variables to free memory 
            del inputs_data, output_data 
        # Create results directory if it doesn't exist
        # First compute SVD to get singular values
        mean_in = np.mean(per_batch_in)
        mean_out = np.mean(per_batch_out)
        per_batch_in -= mean_in
        per_batch_out -= mean_out
        U_temp, S_temp, _ = randomized_svd(per_batch_in, n_components=min(per_batch_in.shape[0], per_batch_in.shape[1], 500), n_iter=2, random_state=42)
        m, n = per_batch_in.shape
        message = f"Dataset: {dataset}, Field Index: {field_idx}, per_batch_in_mean{mean_in}, per_batch_out_mean{mean_out}\n"
        with open("/home/david/My_Data/store_mean.txt", "a") as file:
            file.write(f"{message}\n")

        estimated_rank = optimal_rank(m, n, S_temp)
        if estimated_rank < 1:
            estimated_rank = 1
        print("Estimated Rank: ", estimated_rank)
        U, S, Vt = randomized_svd(per_batch_in, n_components=estimated_rank, n_iter=2, random_state=42)
        U_out, S_out, Vt_out = randomized_svd(per_batch_out, n_components=estimated_rank, n_iter=2, random_state=42)   
        scipy.io.savemat(f"/home/david/My_Data/output_matrix_rank_{estimated_rank}_{dataset}_field_{field_idx}.mat", {f"Matrix_1_{dataset}_in_U{field_idx}": U, f"Matrix_1_{dataset}_in_S{field_idx}": S, f"Matrix_1_{dataset}_in_Vt{field_idx}": Vt, f"Matrix_1_{dataset}_out_U{field_idx}" : U_out, f"Matrix_1_{dataset}_out_S{field_idx}": S_out, f"Matrix_1_{dataset}_out_Vt{field_idx}": Vt_out})
        del per_batch_in, per_batch_out, estimated_rank
    print("Field-wise global SVD complete.")


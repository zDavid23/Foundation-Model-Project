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
import scipy.io
trainset = WellDataset( well_base_path="E:\Datasets\datasets", well_dataset_name="active_matter", well_split_name="train")
train_loader = DataLoader(trainset)
k = 20
count = 0
results_dir = "E:"
per_batch_in = np.zeros((4, 14000, 256*256), dtype = np.uint8) # Reset for each batch
per_batch_out = np.zeros((4, 14000, 256*256), dtype=np.uint8) # Reset for each batch
count = 0
for batch in train_loader: 
    count += 1 
    print(f"Processing batch {count}") 
    for i in range(4): # Process input field 
        input_field = batch["input_fields"][:, :, :, :, i] 
        inputs_data = input_field.reshape(input_field.shape[0] * input_field.shape[1], -1) # Process output field 
        output_field = batch["output_fields"][:, :, :, :, i] 
        output_data = output_field.reshape(output_field.shape[0] * output_field.shape[1], -1) 
        per_batch_in[i, count-1, :] = inputs_data 
        per_batch_out[i, count-1, :] = output_data # Clear intermediate variables to free memory 
        del inputs_data, output_data 
# Create results directory if it doesn't exist
for field_idx in range(4): # Load all batches for the current field 
    scipy.io.savemat(f"output_matrix_{field_idx}.mat", {f"Matrix_1_active_matter_in_{field_idx}": per_batch_in[field_idx, :, :], f"Matrix_1_active_matter_out_{field_idx}" : per_batch_out[field_idx,:,:]})
print("Field-wise global SVD complete.")

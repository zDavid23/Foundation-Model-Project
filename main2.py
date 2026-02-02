import h5py
import torch
import the_well
from torch.utils.data import DataLoader
import numpy as np
import os
import pickle
from the_well.data import WellDataset
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import dask.array as da
import matplotlib.pyplot as plt
from sklearn.utils.extmath import randomized_svd
import scipy.io
import plotly.graph_objects as go
import scipy.io as sio
done = ["MHD_64", "gray_scott_reaction_diffusion", "post_neutron_star_merger", "viscoelastic_instability", "turbulent_radiative_layer_2D", "helmholtz_staircase", "planetswe", "supernova_explosion_64"]
print(len(done))

for dataset in done:
    print(f"Processing dataset: {dataset}", flush=True)
    combined_matrix_in = None
    combined_matrix_out = None
    count = 0
    print(sorted(os.listdir(f"/home/david/My_Data/{dataset}/"), key=lambda x: int(x.split('_')[-1].split('.')[0])))
    # Reconstruct and concatenate all fields
    for idx, filename in enumerate(sorted(os.listdir(f"/home/david/My_Data/{dataset}/"), key=lambda x: int(x.split('_')[-1].split('.')[0]))):        
        count += 1
        print("Processing: ", idx)
        full_path = os.path.join(f"/home/david/My_Data/{dataset}/", filename)
        with h5py.File(full_path, 'r') as f:
            mat_contents = {key: np.array(f[key]) for key in f.keys()}

        # Reconstruct input and output matrices for this field (these are already centered from local SVD)
        matrix_in = mat_contents[f"Matrix_1_{dataset}_in_U{idx}"] @ np.diag(mat_contents[f"Matrix_1_{dataset}_in_S{idx}"].flatten()) @ mat_contents[f"Matrix_1_{dataset}_in_Vt{idx}"]
        matrix_out = mat_contents[f"Matrix_1_{dataset}_out_U{idx}"] @ np.diag(mat_contents[f"Matrix_1_{dataset}_out_S{idx}"].flatten()) @ mat_contents[f"Matrix_1_{dataset}_out_Vt{idx}"]
        
        # Store field-wise means for later reconstruction
        mean_in_field = mat_contents[f"mean_in_{idx}"].item()
        mean_out_field = mat_contents[f"mean_out_{idx}"].item()
        
        if idx == 0:
            field_means_in = [mean_in_field]
            field_means_out = [mean_out_field]
        else:
            field_means_in.append(mean_in_field)
            field_means_out.append(mean_out_field)
        if count == 1:
            new_array_in = np.zeros(mat_contents[f"Matrix_1_{dataset}_shape"])
            new_array_out = np.zeros(mat_contents[f"Matrix_1_{dataset}_shape"])
        if len(new_array_in.shape) == 5:
            new_array_in[:,:,:, :, idx] = np.reshape(matrix_in + mean_in_field, new_array_in[:,:,:, :, idx].shape)
            new_array_out[:,:,:, :, idx] = np.reshape(matrix_out + mean_out_field, new_array_out[:,:,:, :, idx].shape)
        else:
            new_array_in[:,:, :, idx] = np.reshape(matrix_in + mean_in_field, new_array_in[:,:, :, idx].shape)
            new_array_out[:,:, :, idx] = np.reshape(matrix_out + mean_out_field, new_array_out[:,:, :, idx].shape)
    combined_matrix_in = new_array_in
    combined_matrix_out = new_array_out
    
    print(f"Combined matrix shape: {combined_matrix_in.shape}", flush=True)
    
    # Save combined dataset without additional compression
    if not os.path.exists("/home/david/final_datasets/"):
        os.makedirs("/home/david/final_datasets/")
    with h5py.File(f"/home/david/final_datasets/combined_dataset_{dataset}.mat", 'w') as hf:
        hf.create_dataset('combined_matrix_in', data=combined_matrix_in)
        hf.create_dataset('combined_matrix_out', data=combined_matrix_out)
    
    print(f"Combined dataset saved for {dataset}", flush=True)
    del combined_matrix_in, combined_matrix_out

            
            

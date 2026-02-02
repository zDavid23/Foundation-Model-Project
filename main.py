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
import h5py
import plotly.graph_objects as go
done = "active_matter", "MHD_64","supernova_explosion_64", "gray_scott_reaction_diffusion", "post_neutron_star_merger", "viscoelastic_instability", "turbulent_radiative_layer_2D", "helmholtz_staircase", "planetswe",
names_of_datasets_top_run = ["rayleigh_benard","turbulence_gravity_cooling", "rayleigh_benard_uniform", "rayleigh_taylor_instability","shear_flow","turbulence_gravity_cooling", "supernova_explosion_128", "turbulent_radiative_layer_3D", "acoustic_scattering_discontinuous","acoustic_scattering_inclusions", "convective_envelope_rsg", "acoustic_scattering_maze"]
print(len(names_of_datasets_top_run))

def var_optimal_rank(singular_values, total_variance):
    variance_captured = 0
    for rank in range(len(singular_values)):
        variance_captured = sum([j**2 for j in singular_values[:rank]])
        percent_var_captured = variance_captured / total_variance
        if percent_var_captured >= 0.99:
            return rank + 1
    return len(singular_values)
for dataset in names_of_datasets_top_run:
    print(f"Processing dataset: {dataset}", flush=True)
    trainset = WellDataset(well_base_path="/media/david/USB-HDD/Datasets/datasets", well_dataset_name=f"{dataset}", well_split_name="train")
    train_loader = DataLoader(trainset)
    results_dir = "/media/david/USB-HDD/"
    temp = next(iter(train_loader))["input_fields"].shape
    
    for field_idx in range(next(iter(train_loader))["input_fields"].shape[-1]): 
        all_inputs = []
        all_outputs = []
        
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx == 0:
                shape_in = batch["input_fields"].shape
                with open("/home/david/My_Data/dataset_info.txt", "a") as file:
                    file.write(f"Dataset: {dataset}, Field Index: {field_idx}, Input Shape: {batch['input_fields'].shape}, Output Shape: {batch['output_fields'].shape}\n")
                
            print(f"Processing batch {batch_idx + 1}", flush=True)
            
            # Extract the specific field
            if len(batch["input_fields"].shape) == 6:
                input_field = (batch["input_fields"][:, :, :, :, :, field_idx])
                output_field = (batch["output_fields"][:, :, :, :, :, field_idx])
            else:
                if batch_idx == 0:
                    plt.imshow(batch["input_fields"][0, 0, :, :, field_idx].cpu(), cmap='viridis')
                    if not os.path.exists(f"/home/david/My_Data/photos/{dataset}/"):
                        os.makedirs(f"/home/david/My_Data/photos/{dataset}/")
                    plt.savefig(f"/home/david/My_Data/photos/{dataset}/sample_input_field_{dataset}_field_{field_idx}.png")
                input_field = batch["input_fields"][:, :, :, :, field_idx]
                output_field = batch["output_fields"][:, :, :, :, field_idx]
            
            # Reshape: (batch_size, time_steps, spatial...) -> (batch_size * time_steps, spatial_flattened)
            inputs_data = input_field.reshape(input_field.shape[0] * input_field.shape[1], -1)
            output_data = output_field.reshape(output_field.shape[0] * output_field.shape[1], -1)
            
            # Append all samples from this batch
            all_inputs.append(inputs_data)
            all_outputs.append(output_data)
            
            del inputs_data, output_data, input_field, output_field
        
        # Concatenate all batches into single matrices
        per_batch_in = np.vstack(all_inputs)
        shape = [per_batch_in.shape[0]] + list(shape_in[2:])
        per_batch_out = np.vstack(all_outputs)
        del all_inputs, all_outputs
        
        print(f"Final data matrix: {per_batch_in.shape} (total snapshots × spatial points)", flush=True)
        
        # Create results directory if it doesn't exist
        # First compute SVD to get singular values
        mean_in = np.mean(per_batch_in)
        mean_out = np.mean(per_batch_out)
        per_batch_in -= mean_in
        per_batch_out -= mean_out
        # Compute true total variance (fast - no SVD needed)
        total_variance_in = np.linalg.norm(per_batch_in, 'fro')**2
        total_variance_out = np.linalg.norm(per_batch_out, 'fro')**2
        
        # Start with conservative estimate
        n_comp = min(per_batch_in.shape[0], per_batch_in.shape[1], 500)
        U_temp, S_temp, _ = randomized_svd(per_batch_in, n_components=n_comp, n_iter=9, random_state=42)
        n_comp1 = min(per_batch_out.shape[0], per_batch_out.shape[1], 500)
        U_temp1, S_temp1, _ = randomized_svd(per_batch_out, n_components=n_comp1, n_iter=9, random_state=42)    
        
        
        # Check if we captured 99% of TRUE total variance
        variance_captured = np.sum(S_temp**2)
        variance_captured1 = np.sum(S_temp1**2)
        variance_ratio = variance_captured / total_variance_in
        variance_ratio1 = variance_captured1 / total_variance_out
        
        # If not enough, incrementally increase
        while variance_ratio < 0.99 and n_comp < min(per_batch_in.shape[0], per_batch_in.shape[1]):
            n_comp = min(n_comp + 500, per_batch_in.shape[0], per_batch_in.shape[1])
            print(f"Increasing components to {n_comp} (current variance: {variance_ratio*100:.2f}%)", flush=True)
            U_temp, S_temp, _ = randomized_svd(per_batch_in, n_components=n_comp, n_iter=9, random_state=42)
            variance_captured = np.sum(S_temp**2)
            variance_ratio = variance_captured / total_variance_in
        while variance_ratio1 < 0.99 and n_comp1 < min(per_batch_out.shape[0], per_batch_out.shape[1]):
            n_comp1 = min(n_comp1 + 500, per_batch_out.shape[0], per_batch_out.shape[1])
            print(f"Increasing components to {n_comp1} (current variance: {variance_ratio1*100:.2f}%)", flush=True)
            U_temp1, S_temp1, _ = randomized_svd(per_batch_out, n_components=n_comp1, n_iter=9, random_state=42)
            variance_captured1 = np.sum(S_temp1**2)
            variance_ratio1 = variance_captured1 / total_variance_out

        message = f"Dataset: {dataset}, Field Index: {field_idx}, per_batch_in_mean{mean_in}, per_batch_out_mean{mean_out}\n"
        with open("/home/david/My_Data/store_mean.txt", "a") as file:
            file.write(f"{message}\n")
                
        estimated_rank = max(var_optimal_rank(S_temp, total_variance_in), var_optimal_rank(S_temp1, total_variance_out))
        print("Estimated Rank: ", estimated_rank, flush=True)
        with open(f"/home/david/My_Data/estimated_rank_{dataset}.txt", "a") as f:
            f.write("rank: " + str(estimated_rank) + " field: " + str(field_idx) + "\n")
        if not os.path.exists(f"/home/david/My_Data/{dataset}/"):
            os.makedirs(f"/home/david/My_Data/{dataset}/")
        U, S, Vt = randomized_svd(per_batch_in, n_components=estimated_rank, n_iter=9, random_state=42)
        U_out, S_out, Vt_out = randomized_svd(per_batch_out, n_components=estimated_rank, n_iter=9, random_state=42)   
        
        # Save as MATLAB v7.3 (HDF5 format) for large files
        filename = f"/home/david/My_Data/{dataset}/output_matrix_{dataset}_field_{field_idx}.mat"
        with h5py.File(filename, 'w') as f:
            f.create_dataset(f"Matrix_1_{dataset}_in_U{field_idx}", data=U)
            f.create_dataset(f"Matrix_1_{dataset}_in_S{field_idx}", data=S)
            f.create_dataset(f"Matrix_1_{dataset}_in_Vt{field_idx}", data=Vt)
            f.create_dataset(f"Matrix_1_{dataset}_out_U{field_idx}", data=U_out)
            f.create_dataset(f"Matrix_1_{dataset}_out_S{field_idx}", data=S_out)
            f.create_dataset(f"Matrix_1_{dataset}_out_Vt{field_idx}", data=Vt_out)
            f.create_dataset(f"mean_in_{field_idx}", data=mean_in)
            f.create_dataset(f"mean_out_{field_idx}", data=mean_out)
            f.create_dataset(f"Matrix_1_{dataset}_shape", data=np.array(shape))
        
        del per_batch_in, per_batch_out, estimated_rank, n_comp, n_comp1, U, S, Vt, U_out, S_out, Vt_out, mean_in, mean_out, U_temp, S_temp, U_temp1, S_temp1, variance_captured, variance_captured1, variance_ratio, variance_ratio1, total_variance_in, total_variance_out
    print("Field-wise global SVD complete.", flush=True)

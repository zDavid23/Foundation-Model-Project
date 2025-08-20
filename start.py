from the_well.data import WellDataset
from torch.utils.data import DataLoader
import numpy as np
import os
import pickle
trainset = WellDataset(
    well_base_path="path//to//base//datasets",
    well_dataset_name="active_matter",
    well_split_name="train"
)
train_loader = DataLoader(trainset)

k = 10
count = 0
results_dir = "D:\\New folder"
os.makedirs(results_dir, exist_ok=True)

for batch in train_loader:
    # Process input field
    input_field = batch["input_fields"]
    inputs_data = input_field.numpy().reshape(input_field.shape[0] * input_field.shape[1], -1)
    
    # Process output field  
    output_field = batch["output_fields"]
    output_data = output_field.numpy().reshape(output_field.shape[0] * output_field.shape[1], -1)
    
    # Perform SVD for input and output data separately
    U_in, s_in, V_in = np.linalg.svd(inputs_data, full_matrices=False)
    U_out, s_out, V_out = np.linalg.svd(output_data, full_matrices=False)
    
    # Reconstruct the low-rank approximation for this field
    batch_reconstructed_inputs = U_in[:, :k] @ np.diag(s_in[:k]) @ V_in[:k, :]
    batch_reconstructed_outputs = U_out[:, :k] @ np.diag(s_out[:k]) @ V_out[:k, :]
    
    # Clear intermediate variables to free memory
    del U_in, s_in, V_in, U_out, s_out, V_out, inputs_data, output_data
    
    # Save batch results to disk
    np.save(os.path.join(results_dir, f"batch_{count}_inputs.npy"), batch_reconstructed_inputs)
    np.save(os.path.join(results_dir, f"batch_{count}_outputs.npy"), batch_reconstructed_outputs)
    
    # Clear batch data from memory
    del batch_reconstructed_inputs, batch_reconstructed_outputs
    
    # Increment count for next batch
    count += 1


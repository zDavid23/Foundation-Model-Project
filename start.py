from the_well.data import WellDataset
from torch.utils.data import DataLoader
import numpy as np
import os
import pickle
trainset = WellDataset(
    well_base_path="path\\to\\base\\datasets",
    well_dataset_name="active_matter",
    well_split_name="train"
)
train_loader = DataLoader(trainset)

k = 20
count = 0

results_dir = "svd_results"
os.makedirs(results_dir, exist_ok=True)

for batch in train_loader:
    count += 1
    per_field_in = []  # Reset for each batch
    per_field_out = []  # Reset for each batch
    
    for i in range(11):
        # Process input field
        input_field = batch["input_fields"][:, :, :, :, i]
        inputs_data = input_field.reshape(input_field.shape[0] * input_field.shape[1], -1)
        
        # Process output field  
        output_field = batch["output_fields"][:, :, :, :, i]
        output_data = output_field.reshape(output_field.shape[0] * output_field.shape[1], -1)
        
        # Perform SVD for input and output data separately
        U_in, s_in, V_in = np.linalg.svd(inputs_data, full_matrices=False)
        U_out, s_out, V_out = np.linalg.svd(output_data, full_matrices=False)
        
        # Reconstruct the low-rank approximation for this field
        reconstructed_input = U_in[:, :k] @ np.diag(s_in[:k]) @ V_in[:k, :]
        reconstructed_output = U_out[:, :k] @ np.diag(s_out[:k]) @ V_out[:k, :]
        
        per_field_in.append(reconstructed_input)
        per_field_out.append(reconstructed_output)
        
        # Clear intermediate variables to free memory
        del U_in, s_in, V_in, U_out, s_out, V_out, inputs_data, output_data
    
    # Stack fields for this batch
    batch_reconstructed_inputs = np.stack(per_field_in, axis=0)
    batch_reconstructed_outputs = np.stack(per_field_out, axis=0)
    
    # Save batch results to disk
    np.save(os.path.join(results_dir, f"batch_{count}_inputs.npy"), batch_reconstructed_inputs)
    np.save(os.path.join(results_dir, f"batch_{count}_outputs.npy"), batch_reconstructed_outputs)
    
    # Clear batch data from memory
    del batch_reconstructed_inputs, batch_reconstructed_outputs, per_field_in, per_field_out
    


# Create results directory if it doesn't exist
field_svd_results = {}

for field_idx in range(11):
    # Load all batches for the current field
    field_inputs = []
    field_outputs = []
    
    for batch_idx in range(1, count + 1):
        batch_inputs = np.load(os.path.join(results_dir, f"batch_{batch_idx}_inputs.npy"))
        batch_outputs = np.load(os.path.join(results_dir, f"batch_{batch_idx}_outputs.npy"))
        
        field_inputs.append(batch_inputs[field_idx])
        field_outputs.append(batch_outputs[field_idx])
    
    # Concatenate all batches for the current field
    field_inputs = np.concatenate(field_inputs, axis=0)
    field_outputs = np.concatenate(field_outputs, axis=0)
    
    # Perform SVD for the entire field
    U_in, s_in, V_in = np.linalg.svd(field_inputs, full_matrices=False)
    U_out, s_out, V_out = np.linalg.svd(field_outputs, full_matrices=False)
    
    # Save SVD results for the field
    field_svd_results[field_idx] = {
        "U_in": U_in[:, :k],
        "s_in": s_in[:k],
        "V_in": V_in[:k, :],
        "U_out": U_out[:, :k],
        "s_out": s_out[:k],
        "V_out": V_out[:k, :]
    }
    
    # Clear intermediate variables to free memory
    del field_inputs, field_outputs, U_in, s_in, V_in, U_out, s_out, V_out

print("Field-wise global SVD complete.")

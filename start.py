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
data_matrix = []
U_list = []
s_list = []
V_list = []

U_out_list = []
s_out_list = []
V_out_list = []

k = 20
count = 0
reconstructed_inputs = []
reconstructed_outputs = []

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
    
    # Stack fields for this batch
    reconstructed_inputs.append(np.stack(per_field_in, axis=0))
    reconstructed_outputs.append(np.stack(per_field_out, axis=0))
    if count == 500:
        break

# Method 1: Field-wise global SVD to maintain physical meaning
print("Performing field-wise global SVD...")

# Create results directory if it doesn't exist
results_dir = "svd_results"
os.makedirs(results_dir, exist_ok=True)

# Store results for each field
field_svd_results = {}

for field_idx in range(11):
    print(f"Processing global SVD for field {field_idx}")
    
    # Extract this field from all batches
    field_data_in = [batch_data[field_idx] for batch_data in reconstructed_inputs]
    field_data_out = [batch_data[field_idx] for batch_data in reconstructed_outputs]
    
    # Concatenate this field across batches
    field_combined_in = np.concatenate(field_data_in, axis=0)
    field_combined_out = np.concatenate(field_data_out, axis=0)
    
    print(f"Field {field_idx} - Input shape: {field_combined_in.shape}, Output shape: {field_combined_out.shape}")
    
    # Global SVD for this field
    U_global_field_in, s_global_field_in, V_global_field_in = np.linalg.svd(field_combined_in, full_matrices=False)
    U_global_field_out, s_global_field_out, V_global_field_out = np.linalg.svd(field_combined_out, full_matrices=False)
    
    print(f"Field {field_idx} - Global SVD shapes - Input U: {U_global_field_in.shape}, s: {s_global_field_in.shape}, V: {V_global_field_in.shape}")
    print(f"Field {field_idx} - Global SVD shapes - Output U: {U_global_field_out.shape}, s: {s_global_field_out.shape}, V: {V_global_field_out.shape}")
    
    # Store results
    field_svd_results[field_idx] = {
        'input': {
            'U': U_global_field_in,
            's': s_global_field_in,
            'V': V_global_field_in,
            'data': field_combined_in
        },
        'output': {
            'U': U_global_field_out,
            's': s_global_field_out,
            'V': V_global_field_out,
            'data': field_combined_out
        }
    }
    
    # Save individual field results
    np.save(os.path.join(results_dir, f"active_matter_field_{field_idx}_U_global_input.npy"), U_global_field_in)
    np.save(os.path.join(results_dir, f"active_matter_field_{field_idx}_s_global_input.npy"), s_global_field_in)
    np.save(os.path.join(results_dir, f"active_matter_field_{field_idx}_V_global_input.npy"), V_global_field_in)
    
    np.save(os.path.join(results_dir, f"active_matter_field_{field_idx}_U_global_output.npy"), U_global_field_out)
    np.save(os.path.join(results_dir, f"active_matter_field_{field_idx}_s_global_output.npy"), s_global_field_out)
    np.save(os.path.join(results_dir, f"active_matter_field_{field_idx}_V_global_output.npy"), V_global_field_out)
    
    # Save the combined field data
    np.save(os.path.join(results_dir, f"active_matter_field_{field_idx}_combined_input.npy"), field_combined_in)
    np.save(os.path.join(results_dir, f"active_matter_field_{field_idx}_combined_output.npy"), field_combined_out)

# Save metadata about the analysis
metadata = {
    "k_components": k,
    "num_batches_processed": count,
    "num_fields": 11,
    "field_svd_info": {}
}

# Add field-specific metadata
for field_idx in range(11):
    field_info = field_svd_results[field_idx]
    metadata["field_svd_info"][field_idx] = {
        "input_shape": field_info['input']['data'].shape,
        "output_shape": field_info['output']['data'].shape,
        "input_svd_shapes": {
            "U": field_info['input']['U'].shape,
            "s": field_info['input']['s'].shape,
            "V": field_info['input']['V'].shape
        },
        "output_svd_shapes": {
            "U": field_info['output']['U'].shape,
            "s": field_info['output']['s'].shape,
            "V": field_info['output']['V'].shape
        }
    }

with open(os.path.join(results_dir, "metadata.pkl"), "wb") as f:
    pickle.dump(metadata, f)

print(f"Results saved to '{results_dir}' directory:")
print(f"- Field-wise global SVD components for all 11 fields")
print(f"- Combined reconstructed data for each field")
print(f"- Metadata file with analysis parameters")
print("Done!")



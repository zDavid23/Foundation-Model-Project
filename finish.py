from torch.utils.data import DataLoader
import numpy as np
import os
import pickle
import gc

k = 10
count = 14000

results_dir = "E:\\New folder"
os.makedirs(results_dir, exist_ok=True)

# Create results directory if it doesn't exist
svd_results = {}

# Create results directory if it doesn't exist 
field_svd_results = {}

for field_idx in range(11):
    # Load all batches for the current field
    field_inputs = []
    field_outputs = []
    print(f"Processing field {field_idx + 1} of 11...")
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
print("Field-wise SVD complete.")
with open(os.path.join(results_dir, "field_svd_results.pkl"), "wb") as f:
    pickle.dump(field_svd_results, f)
print("Field-wise global SVD complete.")

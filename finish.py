from the_well.data import WellDataset
from torch.utils.data import DataLoader
import numpy as np
import os
import pickle
import gc

k = 10
count = 750

results_dir = "D:\\New folder"
os.makedirs(results_dir, exist_ok=True)

# Create results directory if it doesn't exist
svd_results = {}

# Process in smaller chunks to manage memory
chunk_size = 100  # Process 100 batches at a time

print("Processing all data...")

# Initialize lists to store all data
all_inputs = []
all_outputs = []

# Process data in chunks
for chunk_start in range(1, count + 1, chunk_size):
    chunk_end = min(chunk_start + chunk_size, count + 1)
    print(f"Loading batches {chunk_start} to {chunk_end - 1}")
    
    chunk_inputs = []
    chunk_outputs = []
    
    for batch_idx in range(chunk_start, chunk_end):
        try:
            # Load with memory mapping to reduce memory usage
            batch_inputs = np.load(os.path.join(results_dir, f"batch_{batch_idx}_inputs.npy"), mmap_mode='r')
            batch_outputs = np.load(os.path.join(results_dir, f"batch_{batch_idx}_outputs.npy"), mmap_mode='r')
            
            # Copy all the data
            chunk_inputs.append(batch_inputs.copy())
            chunk_outputs.append(batch_outputs.copy())
            
            # Clear references to the batch arrays
            del batch_inputs, batch_outputs
            
        except Exception as e:
            print(f"Error loading batch {batch_idx}: {e}")
            continue
    
    # Concatenate chunk data and append to all data
    if chunk_inputs:
        all_inputs.append(np.concatenate(chunk_inputs, axis=0))
        all_outputs.append(np.concatenate(chunk_outputs, axis=0))
    
    # Clear chunk data to free memory
    del chunk_inputs, chunk_outputs
    gc.collect()  # Force garbage collection

# Concatenate all chunks
if all_inputs:
    all_inputs = np.concatenate(all_inputs, axis=0)
    all_outputs = np.concatenate(all_outputs, axis=0)
else:
    print("No data loaded")
    exit()

print("Performing SVD for all data...")
# Perform SVD for all data
U_in, s_in, V_in = np.linalg.svd(all_inputs, full_matrices=False)
U_out, s_out, V_out = np.linalg.svd(all_outputs, full_matrices=False)

# Save SVD results
svd_results = {
    "U_in": U_in[:, :k],
    "s_in": s_in[:k],
    "V_in": V_in[:k, :],
    "all_inputs": all_inputs,
    "all_outputs": all_outputs,
    "U_out": U_out[:, :k],
    "s_out": s_out[:k],
    "V_out": V_out[:k, :]
}

# Clear intermediate variables to free memory
del all_inputs, all_outputs, U_in, s_in, V_in, U_out, s_out, V_out
gc.collect()  # Force garbage collection

print("Data processing complete.")

# Save the final results
print("Saving SVD results...")
with open(os.path.join(results_dir, "svd_results.pkl"), "wb") as f:
    pickle.dump(svd_results, f)

print("Global SVD complete.")

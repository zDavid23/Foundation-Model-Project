from torch.utils.data import DataLoader
import numpy as np
import os
import pickle
import gc

k = 10
count = 13999

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


print("Performing SVD per field (column)...")
svd_results = {
    "inputs": {},
    "outputs": {},
    "all_inputs": all_inputs,
    "all_outputs": all_outputs
}

# SVD per field for inputs
for i in range(all_inputs.shape[1]):
    field_data = all_inputs[:, i].reshape(-1, 1)
    U, s, V = np.linalg.svd(field_data, full_matrices=False)
    svd_results["inputs"][f"field_{i}"] = {
        "U": U[:, :k],
        "s": s[:k],
        "V": V[:k, :]
    }

# SVD per field for outputs
for i in range(all_outputs.shape[1]):
    field_data = all_outputs[:, i].reshape(-1, 1)
    U, s, V = np.linalg.svd(field_data, full_matrices=False)
    svd_results["outputs"][f"field_{i}"] = {
        "U": U[:, :k],
        "s": s[:k],
        "V": V[:k, :]
    }

# Clear intermediate variables to free memory
del all_inputs, all_outputs, U, s, V, field_data
gc.collect()  # Force garbage collection

print("Data processing complete.")

# Save the final results
print("Saving SVD results...")
with open(os.path.join(results_dir, "svd_results_per_field.pkl"), "wb") as f:
    pickle.dump(svd_results, f)

print("Per-field SVD complete.")

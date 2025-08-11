import numpy as np
import h5py

with h5py.File("/Users/davidzoro/Foundation-Model-Project-1/path/to/base/datasets/gray_scott_reaction_diffusion/data/train/gray_scott_reaction_diffusion_maze_F_0.029_k_0.057.hdf5", 'r') as f:
    print("Keys in the file:", list(f.keys()))

"""
U, s, Vh = np.linalg.svd(data)
print("The shape of U", U.shape)
print("The shape of s", s.shape)
print("The shape of Vh", Vh.shape)
"""
import numpy as np
import os

train_dir = '../data_npy/train'

try:
    print(f"Scanning directory: {train_dir}\n")
    files = os.listdir(train_dir)
    
    for filename in files:
        if not filename.endswith('.npy'):
            continue
            
        file_path = os.path.join(train_dir, filename)
        
        try:
            # THE FIX: mmap_mode='r' reads the shape without loading into RAM
            data = np.load(file_path, mmap_mode='r')
            
            print(f"--- File: {filename} ---")
            print(f"Shape: {data.shape}")
            print(f"Data Type: {data.dtype}")
            print("-" * 30)
            
        except ValueError as e:
            # If the array contains python objects (dicts), mmap might fail. 
            # We catch it and warn.
            print(f"--- File: {filename} ---")
            print(f"Could not memory-map. It likely contains a pickled dictionary.")
            print(f"Error details: {e}")
            print("-" * 30)

except FileNotFoundError:
    print(f"Error: Could not find the directory '{train_dir}'.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
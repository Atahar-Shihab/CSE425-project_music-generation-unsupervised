import torch
import os
import numpy as np

def split_dataset():
    print("Loading full dataset...")
    
    # 1. Get the directory where this script lives (.../data/train_test_split)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Go UP two levels to the main project folder
    project_root = os.path.dirname(os.path.dirname(current_dir))
    
    # 3. Correctly route to the processed tensor
    data_path = os.path.join(project_root, "data", "processed", "lakh_tensor.pt")
    
    # Load the massive 15,000-file tensor
    full_data = torch.load(data_path, weights_only=True)
    total_samples = full_data.shape[0]
    print(f"Total sequences loaded: {total_samples}")
    
    # Create indices and shuffle them precisely
    indices = np.random.permutation(total_samples)
    
    # 80% Train, 10% Validation, 10% Test
    train_split = int(0.8 * total_samples)
    val_split = int(0.9 * total_samples)
    
    train_idx = indices[:train_split]
    val_idx = indices[train_split:val_split]
    test_idx = indices[val_split:]
    
    print("Splitting tensors...")
    train_data = full_data[train_idx]
    val_data = full_data[val_idx]
    test_data = full_data[test_idx]
    
    # Save the splits right next to where the script is located
    torch.save(train_data, os.path.join(current_dir, "train.pt"))
    torch.save(val_data, os.path.join(current_dir, "val.pt"))
    torch.save(test_data, os.path.join(current_dir, "test.pt"))
    
    print(f"Success! Data precisely split and saved:")
    print(f"Train: {train_data.shape[0]} samples")
    print(f"Val: {val_data.shape[0]} samples")
    print(f"Test: {test_data.shape[0]} samples")

if __name__ == "__main__":
    split_dataset()
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import sys

# Dynamically set up paths to find the models folder
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

# Import the LSTM Autoencoder class from the models folder
from src.models.autoencoder import LSTMAutoencoder

def train_lstm_ae():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training LSTM Autoencoder on device: {device}")

    # Point to the new TRAINING split!
    data_path = os.path.join(project_root, "data", "train_test_split", "train.pt")
    
    train_data = torch.load(data_path, weights_only=True)
    dataset = TensorDataset(train_data)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    model = LSTMAutoencoder().to(device)
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    epochs = 5 # Optimized for the 15k dataset
    
    model.train()
    for epoch in range(epochs):
        train_loss = 0
        for batch_idx, (data,) in enumerate(dataloader):
            data = data.to(device)
            optimizer.zero_grad()
            
            reconstructed = model(data)
            loss = criterion(reconstructed, data)
            
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            
        print(f"Epoch [{epoch+1}/{epochs}] \t Reconstruction Loss: {train_loss/len(dataloader):.4f}")
        
    save_dir = os.path.join(project_root, "outputs")
    torch.save(model.state_dict(), os.path.join(save_dir, "lstm_ae_model.pt"))
    print("\nTraining Complete! LSTM weights saved.")

if __name__ == "__main__":
    train_lstm_ae()
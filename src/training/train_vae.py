import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

# Import the model and the custom loss function
from src.models.vae import MusicVAE, vae_loss

def train_vae():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training VAE on device: {device}")

    # Point to the precise Training split
    data_path = os.path.join(project_root, "data", "train_test_split", "train.pt")
    
    train_data = torch.load(data_path, weights_only=True)
    dataset = TensorDataset(train_data)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    model = MusicVAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    epochs = 10
    
    model.train()
    for epoch in range(epochs):
        train_loss = 0
        for batch_idx, (data,) in enumerate(dataloader):
            data = data.to(device)
            optimizer.zero_grad()
            
            reconstructed, mu, logvar = model(data)
            loss = vae_loss(reconstructed, data, mu, logvar)
            
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            
        print(f"Epoch [{epoch+1}/{epochs}] \t Average Total Loss: {train_loss/len(dataloader):.4f}")
        
    save_dir = os.path.join(project_root, "outputs")
    torch.save(model.state_dict(), os.path.join(save_dir, "vae_model.pt"))
    print("\nTraining Complete! VAE weights saved.")

if __name__ == "__main__":
    train_vae()
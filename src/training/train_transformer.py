import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import sys
import math

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from src.models.transformer import MusicTransformer

def train_transformer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Transformer on device: {device}")

    data_path = os.path.join(project_root, "data", "train_test_split", "train.pt")
    
    data_tensor = torch.load(data_path, weights_only=True)
    inputs = data_tensor[:, :-1, :]
    targets = data_tensor[:, 1:, :]
    
    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = MusicTransformer().to(device)
    criterion = nn.BCELoss() 
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    epochs = 5 
    
    model.train()
    for epoch in range(epochs):
        train_loss = 0
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            predictions = model(x)
            loss = criterion(predictions, y)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_loss = train_loss/len(dataloader)
        perplexity = math.exp(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}] \t Loss: {avg_loss:.4f} \t Perplexity: {perplexity:.4f}")
        
    save_dir = os.path.join(project_root, "outputs")
    torch.save(model.state_dict(), os.path.join(save_dir, "transformer_model.pt"))
    print("\nTraining Complete! Transformer weights saved.")

if __name__ == "__main__":
    train_transformer()
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os

# ==========================================
# 1. TASK 1: LSTM AUTOENCODER ARCHITECTURE
# ==========================================
class LSTMAutoencoder(nn.Module):
    def __init__(self, feature_dim=88, hidden_dim=256, latent_dim=128):
        super(LSTMAutoencoder, self).__init__()
        self.hidden_dim = hidden_dim
        
        # Encoder: z = f(X)
        self.encoder_lstm = nn.LSTM(input_size=feature_dim, hidden_size=hidden_dim, batch_first=True)
        self.encoder_fc = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder: X_hat = g(z)
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(input_size=hidden_dim, hidden_size=feature_dim, batch_first=True)
        
    def encode(self, x):
        # x shape: (batch, seq_len, features)
        _, (hidden, _) = self.encoder_lstm(x)
        # hidden shape: (1, batch, hidden_dim)
        z = self.encoder_fc(hidden.squeeze(0))
        return z
        
    def decode(self, z, seq_len):
        # Expand latent vector to match sequence length
        z_expanded = self.decoder_fc(z).unsqueeze(1).repeat(1, seq_len, 1)
        out, _ = self.decoder_lstm(z_expanded)
        # Binarize output probability using Sigmoid
        return torch.sigmoid(out)

    def forward(self, x):
        seq_len = x.shape[1]
        z = self.encode(x)
        x_hat = self.decode(z, seq_len)
        return x_hat

# ==========================================
# 2. TRAINING LOOP
# ==========================================
def train_lstm_ae():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training LSTM Autoencoder on device: {device}")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    data_path = os.path.join(project_root, "data", "processed", "lakh_tensor.pt")
    
    data_tensor = torch.load(data_path)
    dataset = TensorDataset(data_tensor)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    model = LSTMAutoencoder().to(device)
    # MSE Loss matches the mathematical requirement: L_AE = ||X - X_hat||^2
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    epochs = 15
    
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
    print("\nTask 1 Complete! LSTM weights saved.")

if __name__ == "__main__":
    train_lstm_ae()
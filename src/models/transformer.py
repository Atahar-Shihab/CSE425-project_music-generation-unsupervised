import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import math

# ==========================================
# 1. TASK 3: TRANSFORMER ARCHITECTURE
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class MusicTransformer(nn.Module):
    def __init__(self, feature_dim=88, d_model=256, nhead=8, num_layers=4):
        super().__init__()
        self.embedding = nn.Linear(feature_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # We use TransformerEncoder with a causal mask to act as an autoregressive decoder
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=512, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc_out = nn.Linear(d_model, feature_dim)
        
    def generate_square_subsequent_mask(self, sz):
        # Prevents the model from looking into the future
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x):
        mask = self.generate_square_subsequent_mask(x.size(1)).to(x.device)
        x = self.embedding(x)
        x = self.pos_encoder(x)
        output = self.transformer(x, mask=mask)
        # Sigmoid bounds the output between 0 and 1
        return torch.sigmoid(self.fc_out(output))

# ==========================================
# 2. TRAINING LOOP & PERPLEXITY METRIC
# ==========================================
def train_transformer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Transformer on device: {device}")

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(project_root, "data", "processed", "lakh_tensor.pt")
    
    data_tensor = torch.load(data_path)
    # Target is the input shifted by 1 timestep for autoregressive training
    inputs = data_tensor[:, :-1, :]
    targets = data_tensor[:, 1:, :]
    
    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = MusicTransformer().to(device)
    criterion = nn.BCELoss() # Approximates Negative Log Likelihood for binary events
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    epochs = 10 
    
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
        # Perplexity = exp(Loss)
        perplexity = math.exp(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}] \t Loss: {avg_loss:.4f} \t Perplexity: {perplexity:.4f}")
        
    save_dir = os.path.join(project_root, "outputs")
    torch.save(model.state_dict(), os.path.join(save_dir, "transformer_model.pt"))
    print("\nTask 3 Complete! Transformer weights saved.")

if __name__ == "__main__":
    train_transformer()
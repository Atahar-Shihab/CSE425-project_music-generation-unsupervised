import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import os

# ==========================================
# 1. MODEL ARCHITECTURE: Variational Autoencoder
# ==========================================
class MusicVAE(nn.Module):
    def __init__(self, sequence_length=128, feature_dim=88, hidden_dim=512, latent_dim=128):
        super(MusicVAE, self).__init__()
        
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.input_dim = sequence_length * feature_dim
        
        # Encoder: Maps X to latent distribution parameters (mu, sigma)
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2_mu = nn.Linear(hidden_dim, latent_dim)      
        self.fc2_logvar = nn.Linear(hidden_dim, latent_dim)  
        
        # Decoder: Maps latent vector z back to reconstructed X
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, self.input_dim)
        
    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc2_mu(h1), self.fc2_logvar(h1)
    
    def reparameterize(self, mu, logvar):
        """ 
        The Reparameterization Trick: z = mu + sigma * epsilon 
        Allows backpropagation through the random sampling process.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        # Sigmoid bounds the output between 0 and 1 (note off / note on)
        return torch.sigmoid(self.fc4(h3))
    
    def forward(self, x):
        x_flat = x.view(-1, self.input_dim)
        mu, logvar = self.encode(x_flat)
        z = self.reparameterize(mu, logvar)
        reconstructed_flat = self.decode(z)
        return reconstructed_flat.view(-1, self.sequence_length, self.feature_dim), mu, logvar

# ==========================================
# 2. LOSS FUNCTION
# ==========================================
def vae_loss_function(reconstructed_x, x, mu, logvar, beta=0.1):
    """
    Computes L_VAE = L_recon + beta * D_KL
    """
    # Reconstruction Loss (Binary Cross Entropy)
    BCE = F.binary_cross_entropy(reconstructed_x, x, reduction='sum')
    
    # KL Divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return BCE + beta * KLD, BCE, KLD

# ==========================================
# 3. TRAINING LOOP
# ==========================================
def train_model():
    # Setup Device (This will automatically use your RTX 4060!)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Dynamically find the processed data tensor
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    data_path = os.path.join(project_root, "data", "processed", "lakh_tensor.pt")
    
    if not os.path.exists(data_path):
        print(f"Error: Could not find {data_path}. Please run midi_parser.py first.")
        return
        
    print("Loading dataset into memory...")
    data_tensor = torch.load(data_path)
    
    # Create DataLoader for batching
    dataset = TensorDataset(data_tensor)
    batch_size = 64
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize Model and push to GPU
    model = MusicVAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    epochs = 25 # You can increase this later for better music quality
    
    print(f"Starting Training for {epochs} Epochs...")
    model.train()
    
    for epoch in range(epochs):
        train_loss = 0
        for batch_idx, (data,) in enumerate(dataloader):
            # Move data to GPU
            data = data.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            reconstructed_batch, mu, logvar = model(data)
            
            # Calculate loss 
            loss, bce, kld = vae_loss_function(reconstructed_batch, data, mu, logvar, beta=0.1)
            
            # Backward pass
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            
        avg_loss = train_loss / len(dataloader.dataset)
        print(f"Epoch [{epoch+1}/{epochs}] \t Average Total Loss: {avg_loss:.4f}")
        
    # Save the trained model weights
    save_dir = os.path.join(project_root, "outputs")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "vae_model.pt")
    torch.save(model.state_dict(), save_path)
    print(f"\nTraining Complete! Model weights saved to: {save_path}")

if __name__ == "__main__":
    train_model()
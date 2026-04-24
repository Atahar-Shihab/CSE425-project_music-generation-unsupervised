import torch
import torch.nn as nn
import torch.nn.functional as F

class MusicVAE(nn.Module):
    def __init__(self, feature_dim=88, hidden_dim=256, latent_dim=128):
        super(MusicVAE, self).__init__()
        
        # Encoder
        self.encoder_lstm = nn.LSTM(feature_dim, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(hidden_dim, feature_dim, batch_first=True)
        
    def encode(self, x):
        _, (hidden, _) = self.encoder_lstm(x)
        hidden = hidden.squeeze(0)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        return mu, logvar
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, z, seq_len=128):
        z_expanded = self.decoder_fc(z).unsqueeze(1).repeat(1, seq_len, 1)
        out, _ = self.decoder_lstm(z_expanded)
        return torch.sigmoid(out)
        
    def forward(self, x):
        seq_len = x.shape[1]
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z, seq_len)
        return reconstructed, mu, logvar

def vae_loss(reconstructed, target, mu, logvar):
    # Reconstruction Loss
    BCE = F.binary_cross_entropy(reconstructed, target, reduction='sum')
    
    # KL Divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return BCE + KLD
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import Config

class MusicVAE(nn.Module):
    def __init__(self):
        super(MusicVAE, self).__init__()
        # Encoder: Compressing (128, 88) -> Latent Space
        self.encoder_lstm = nn.LSTM(Config.FEATURE_DIM, Config.HIDDEN_DIM, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc_mu = nn.Linear(Config.HIDDEN_DIM, Config.LATENT_DIM)
        self.fc_logvar = nn.Linear(Config.HIDDEN_DIM, Config.LATENT_DIM)
        
        # Decoder: Latent Space -> (128, 88)
        self.decoder_fc = nn.Linear(Config.LATENT_DIM, Config.HIDDEN_DIM)
        self.decoder_lstm = nn.LSTM(Config.HIDDEN_DIM, Config.FEATURE_DIM, batch_first=True)
        
    def encode(self, x):
        _, (hidden, _) = self.encoder_lstm(x)
        h = self.dropout(hidden.squeeze(0))
        return self.fc_mu(h), self.fc_logvar(h)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, z, seq_len=Config.SEQ_LEN):
        z_ext = self.decoder_fc(z).unsqueeze(1).repeat(1, seq_len, 1)
        out, _ = self.decoder_lstm(z_ext)
        return torch.sigmoid(out)
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(reconstructed, target, mu, logvar):
    BCE = F.binary_cross_entropy(reconstructed, target, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD
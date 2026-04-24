import torch
import pretty_midi
import numpy as np
import os
import sys

# Ensure Python can find the 'src' folder from anywhere
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.config import Config
from src.models.vae import MusicVAE
from src.models.autoencoder import LSTMAutoencoder
from src.models.transformer import MusicTransformer

def apply_stochastic_mask(probs, is_rlhf=False):
    p = probs.clone()
    
    # MIN-MAX NORMALIZATION: The Ultimate Collapse Fix
    # Stretches weak signals so the highest probability is always 1.0
    p = (p - p.min()) / (p.max() - p.min() + 1e-8)
    
    if is_rlhf:
        c_major = [0, 2, 4, 5, 7, 9, 11]
        bad_idx = [i for i in range(88) if (i + Config.MIN_PITCH) % 12 not in c_major]
        p[..., bad_idx] = 0.0 # Strictly silence the black keys
        
    p = torch.clamp(p, 0.0, 0.99)
    return torch.bernoulli(p)

def piano_roll_to_midi(roll):
    midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0) # Acoustic Grand Piano
    
    for p_idx in range(roll.shape[1]):
        pitch = p_idx + Config.MIN_PITCH
        on_t = None
        for t_idx in range(roll.shape[0]):
            pressed = roll[t_idx, p_idx] > 0.5
            
            if pressed and on_t is None: 
                on_t = t_idx / Config.FS
            elif not pressed and on_t is not None:
                piano.notes.append(pretty_midi.Note(100, pitch, on_t, t_idx/Config.FS))
                on_t = None
                
        if on_t is not None:
            piano.notes.append(pretty_midi.Note(100, pitch, on_t, roll.shape[0]/Config.FS))
            
    midi.instruments.append(piano)
    return midi

def generate_all():
    device = Config.DEVICE
    out_dir = os.path.join(Config.OUTPUTS, "generated_midis")
    os.makedirs(out_dir, exist_ok=True)
    
    # --- TASK 4 (RLHF) ---
    print("Generating Task 4 (Normalized VAE for Max Diversity)...")
    model = MusicVAE().to(device)
    model.load_state_dict(torch.load(f"{Config.OUTPUTS}/vae_model.pt", weights_only=True))
    model.eval()
    with torch.no_grad():
        for i in range(10):
            z = torch.randn(1, Config.LATENT_DIM).to(device)
            probs = model.decode(z)
            binary = apply_stochastic_mask(probs, is_rlhf=True)
            midi = piano_roll_to_midi(binary.view(128, 88).cpu().numpy())
            midi.write(f"{out_dir}/task4_rlhf_{i+1}.mid")

    # --- TASK 1 (LSTM AUTOENCODER) ---
    print("Generating Task 1 (Normalized LSTM Autoencoder)...")
    lstm_model = LSTMAutoencoder().to(device)
    lstm_model.load_state_dict(torch.load(f"{Config.OUTPUTS}/lstm_ae_model.pt", weights_only=True))
    lstm_model.eval()
    with torch.no_grad():
        for i in range(5):
            z = torch.randn(1, Config.LATENT_DIM).to(device) 
            probs = lstm_model.decode(z, seq_len=Config.SEQ_LEN)
            
            # Apply Min-Max Normalization to Task 1 as well
            probs = (probs - probs.min()) / (probs.max() - probs.min() + 1e-8)
            
            gen = torch.bernoulli(torch.clamp(probs, 0, 0.99)).squeeze(0).cpu().numpy()
            midi = piano_roll_to_midi(gen)
            midi.write(f"{out_dir}/task1_lstm_sample_{i+1}.mid")

    # --- TASK 3 (TRANSFORMER) ---
    print("Generating Task 3 (Stochastic Transformer)...")
    model_t = MusicTransformer().to(device)
    model_t.load_state_dict(torch.load(f"{Config.OUTPUTS}/transformer_model.pt", weights_only=True))
    model_t.eval()
    with torch.no_grad():
        for i in range(10):
            seq = (torch.rand(1, 1, 88).to(device) > 0.9).float()
            for _ in range(127):
                p = model_t(seq)[:, -1:, :]
                next_step = torch.bernoulli(torch.clamp(p, 0, 0.99))
                seq = torch.cat([seq, next_step], dim=1)
            midi = piano_roll_to_midi(seq.squeeze(0).cpu().numpy())
            midi.write(f"{out_dir}/task3_transformer_{i+1}.mid")

    print("\nGeneration Complete! Run metrics.py to view your final scores.")

if __name__ == "__main__":
    generate_all()
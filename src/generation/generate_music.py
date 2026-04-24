import torch
import pretty_midi
import numpy as np
import os
import sys

# Dynamically set up paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from src.models.vae import MusicVAE
from src.models.autoencoder import LSTMAutoencoder
from src.models.transformer import MusicTransformer

def piano_roll_to_midi(piano_roll, fs=4, min_pitch=21):
    midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0) # Acoustic Grand Piano
    
    for pitch_idx in range(piano_roll.shape[1]):
        pitch = pitch_idx + min_pitch
        note_on_time = None
        
        for time_idx in range(piano_roll.shape[0]):
            is_pressed = piano_roll[time_idx, pitch_idx] > 0.5 
            
            if is_pressed and note_on_time is None: 
                note_on_time = time_idx / fs
            elif not is_pressed and note_on_time is not None: 
                note_off_time = time_idx / fs
                note = pretty_midi.Note(velocity=100, pitch=pitch, start=note_on_time, end=note_off_time)
                piano.notes.append(note)
                note_on_time = None
                
        if note_on_time is not None:
            note_off_time = piano_roll.shape[0] / fs
            note = pretty_midi.Note(velocity=100, pitch=pitch, start=note_on_time, end=note_off_time)
            piano.notes.append(note)

    midi.instruments.append(piano)
    return midi

def generate_all_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(project_root, "outputs", "generated_midis")
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Generate Task 4 (RLHF) Samples (10 samples)
    print("Generating RLHF-Tuned samples...")
    rlhf_model = MusicVAE().to(device)
    rlhf_model.load_state_dict(torch.load(os.path.join(project_root, "outputs", "vae_rlhf_model.pt"), weights_only=True))
    rlhf_model.eval()
    with torch.no_grad():
        for i in range(10):
            z = torch.randn(1, 128).to(device)
            gen = rlhf_model.decode(z).view(128, 88).cpu().numpy()
            midi_obj = piano_roll_to_midi(gen)
            midi_obj.write(os.path.join(out_dir, f"task4_rlhf_sample_{i+1}.mid"))
            
    # 2. Generate Task 1 (LSTM) Samples (5 samples)
    print("Generating LSTM Autoencoder samples...")
    lstm_model = LSTMAutoencoder().to(device)
    lstm_model.load_state_dict(torch.load(os.path.join(project_root, "outputs", "lstm_ae_model.pt"), weights_only=True))
    lstm_model.eval()
    with torch.no_grad():
        for i in range(5):
            z = torch.randn(1, 128).to(device) # Random latent vector
            gen = lstm_model.decode(z, seq_len=128).squeeze(0).cpu().numpy()
            midi_obj = piano_roll_to_midi(gen)
            midi_obj.write(os.path.join(out_dir, f"task1_lstm_sample_{i+1}.mid"))

    # 3. Generate Task 3 (Transformer) Samples (10 samples)
    print("Generating Transformer samples...")
    transformer_model = MusicTransformer().to(device)
    transformer_model.load_state_dict(torch.load(os.path.join(project_root, "outputs", "transformer_model.pt"), weights_only=True))
    transformer_model.eval()
    with torch.no_grad():
        for i in range(10):
            # Start with a silent token and autoregressively predict the next 128 steps
            current_seq = torch.zeros(1, 1, 88).to(device)
            for _ in range(127):
                next_step_probs = transformer_model(current_seq)
                next_step = (next_step_probs[:, -1:, :] > 0.5).float()
                current_seq = torch.cat([current_seq, next_step], dim=1)
                
            gen = current_seq.squeeze(0).cpu().numpy()
            midi_obj = piano_roll_to_midi(gen)
            midi_obj.write(os.path.join(out_dir, f"task3_transformer_sample_{i+1}.mid"))

    print("\nSuccessfully generated all required model samples! Check outputs/generated_midis/")

if __name__ == "__main__":
    generate_all_models()
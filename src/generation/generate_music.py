import torch
import pretty_midi
import numpy as np
import os
import sys

# Dynamically set up paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root) # Allow importing from src

# Import the model architecture you just trained
from src.models.vae import MusicVAE

def piano_roll_to_midi(piano_roll, fs=4, min_pitch=21):
    """
    Converts a 2D numpy array (time_steps, 88_keys) back into a playable .mid file.
    """
    midi = pretty_midi.PrettyMIDI()
    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program)
    
    # Iterate through time steps and pitches to find notes
    for pitch_idx in range(piano_roll.shape[1]):
        pitch = pitch_idx + min_pitch
        note_on_time = None
        
        for time_idx in range(piano_roll.shape[0]):
            # The VAE outputs numbers between 0 and 1. We treat anything > 0.5 as a pressed key.
            is_pressed = piano_roll[time_idx, pitch_idx] > 0.5 
            
            if is_pressed and note_on_time is None: 
                # Note just started
                note_on_time = time_idx / fs
            elif not is_pressed and note_on_time is not None: 
                # Note just ended
                note_off_time = time_idx / fs
                note = pretty_midi.Note(velocity=100, pitch=pitch, start=note_on_time, end=note_off_time)
                piano.notes.append(note)
                note_on_time = None # Reset
                
        # Catch any notes that are still held down at the very end of the sequence
        if note_on_time is not None:
            note_off_time = piano_roll.shape[0] / fs
            note = pretty_midi.Note(velocity=100, pitch=pitch, start=note_on_time, end=note_off_time)
            piano.notes.append(note)

    midi.instruments.append(piano)
    return midi

def generate_samples(num_samples=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the model and load your trained weights
    model = MusicVAE().to(device)
    model_path = os.path.join(project_root, "outputs", "vae_model.pt")
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval() # Set model to evaluation mode
    
    out_dir = os.path.join(project_root, "outputs", "generated_midis")
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Generating {num_samples} new AI music tracks...")
    
    with torch.no_grad():
        for i in range(num_samples):
            # Sample random noise from the normal distribution (Latent Space)
            # z = mu + sigma * epsilon, but here we just sample standard normal N(0, I)
            z = torch.randn(1, 128).to(device) 
            
            # Decode the noise into a piano roll
            generated_tensor = model.decode(z)
            generated_tensor = generated_tensor.view(128, 88).cpu().numpy()
            
            # Convert to MIDI and save
            midi_obj = piano_roll_to_midi(generated_tensor)
            file_name = os.path.join(out_dir, f"vae_generated_sample_{i+1}.mid")
            midi_obj.write(file_name)
            print(f"Saved: {file_name}")

if __name__ == "__main__":
    generate_samples(num_samples=8)
    print("\nCheck your 'outputs/generated_midis/' folder to listen to your AI music!")
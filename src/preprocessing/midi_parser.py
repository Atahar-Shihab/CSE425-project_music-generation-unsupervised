import pretty_midi
import numpy as np
import torch
import os
import sys
import warnings

# Ensure Python can find the 'src' folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.config import Config

# Silence the spammy warnings from poorly formatted MIDI files
warnings.filterwarnings("ignore")

class MidiProcessor:
    def __init__(self):
        self.fs = Config.FS
        self.seq_len = Config.SEQ_LEN

    def midi_to_piano_roll(self, file_path):
        try:
            midi_data = pretty_midi.PrettyMIDI(file_path)
            piano_roll = midi_data.get_piano_roll(fs=self.fs)
            
            # Slice to only the 88 piano keys
            piano_roll = piano_roll[Config.MIN_PITCH:Config.MAX_PITCH+1, :]
            piano_roll = piano_roll.T
            
            # Convert to binary notes
            piano_roll = (piano_roll > 0).astype(np.float32)
            return piano_roll
            
        except Exception:
            # Catch standard corrupted file errors
            return None
        except:
            # BULLETPROOF: Catch literally any fatal crash, memory error, 
            # or frozen core exception and skip the file.
            return None

    def create_dataset(self, limit=15000):
        all_sequences = []
        clean_count = 0
        scanned_count = 0
        
        print(f"Starting Bulletproof Parsing from {Config.DATA_RAW}...")
        
        for root, _, files in os.walk(Config.DATA_RAW):
            for file in files:
                if file.endswith((".mid", ".midi")):
                    scanned_count += 1
                    path = os.path.join(root, file)
                    
                    roll = self.midi_to_piano_roll(path)
                    
                    if roll is not None and len(roll) >= self.seq_len:
                        # Slice the first seq_len steps
                        all_sequences.append(roll[:self.seq_len, :])
                        clean_count += 1
                    
                    # LIVE PROGRESS UPDATE: Prints every 500 files so you know it's working
                    if scanned_count % 500 == 0:
                        print(f"Scanned {scanned_count} files... Extracted {clean_count} clean sequences.")
                    
                    if clean_count >= limit:
                        break
            if clean_count >= limit:
                break
        
        # Save the massive tensor
        dataset_tensor = torch.tensor(np.array(all_sequences))
        os.makedirs(Config.DATA_PROCESSED, exist_ok=True)
        save_path = os.path.join(Config.DATA_PROCESSED, "lakh_tensor.pt")
        
        torch.save(dataset_tensor, save_path)
        print(f"\nSUCCESS! Saved {clean_count} pristine sequences to {save_path}")
        return dataset_tensor

if __name__ == "__main__":
    processor = MidiProcessor()
    processor.create_dataset()
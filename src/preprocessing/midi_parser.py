import pretty_midi
import numpy as np
import os
import glob
import torch

def process_midi_file(file_path, seq_length=128, min_pitch=21, max_pitch=108):
    """
    Reads a .mid file and converts it to a binary piano roll representation.
    min_pitch=21 (A0) and max_pitch=108 (C8) represents the 88 keys on a standard piano.
    """
    try:
        # Load the MIDI file
        midi_data = pretty_midi.PrettyMIDI(file_path)
        
        # Get piano roll. fs=4 means we sample the music 4 times per second.
        # Shape will be (128 possible MIDI pitches, time_steps)
        piano_roll = midi_data.get_piano_roll(fs=4)
        
        # Crop the data to only include the 88 standard piano keys
        piano_roll = piano_roll[min_pitch:max_pitch + 1, :]
        
        # Binarize: If a note is played (velocity > 0), set to 1, else 0
        piano_roll[piano_roll > 0] = 1
        
        # Transpose so time is the first dimension: (time_steps, 88_keys)
        piano_roll = piano_roll.T
        
        # Slice the continuous song into shorter, fixed-length sequences (e.g., 128 steps)
        # This matches the 'sequence_length' expected by our Neural Network
        sequences = []
        for i in range(0, piano_roll.shape[0] - seq_length, seq_length):
            chunk = piano_roll[i:i + seq_length, :]
            # Only keep chunks that actually have music in them (ignore long silences)
            if np.sum(chunk) > 10: 
                sequences.append(chunk)
                
        return sequences
        
    except Exception as e:
        # Some MIDI files in LMD might be corrupted, we just gracefully skip them
        return None

def create_dataset(lmd_directory, max_files=500):
    """
    Searches through the Lakh MIDI dataset folder, parses files, and stacks them into a single dataset.
    """
    # Get the absolute path dynamically so it works from anywhere in VS Code
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    full_lmd_dir = os.path.join(project_root, lmd_directory)
    
    print(f"Searching for MIDI files in: {full_lmd_dir}")
    midi_files = glob.glob(os.path.join(full_lmd_dir, '**', '*.mid'), recursive=True)
    
    print(f"Found {len(midi_files)} files. Processing up to {max_files}...")
    all_sequences = []
    files_processed = 0
    
    for f in midi_files:
        if files_processed >= max_files:
            break
            
        seqs = process_midi_file(f)
        if seqs is not None and len(seqs) > 0:
            all_sequences.extend(seqs)
            files_processed += 1
            if files_processed % 50 == 0:
                print(f"Processed {files_processed} files...")
                
    if len(all_sequences) > 0:
        # Stack all sequences into one massive 3D numpy array
        final_data = np.array(all_sequences)
        print(f"\nSuccess! Total dataset shape: {final_data.shape}")
        print(f"(Number of Sequences, Sequence Length, Number of Keys)")
        return final_data
    else:
        print("No sequences could be extracted. Check your folder path.")
        return np.array([])

if __name__ == "__main__":
    # The relative path to your raw MIDI data
    DATA_DIR = "data/raw_midi/lmd_full" 
    
    # Start small! Process just 100 files to test it out quickly.
    dataset_array = create_dataset(DATA_DIR, max_files=15000)
    
    if dataset_array.size > 0:
        # Convert the NumPy array to a PyTorch tensor
        tensor_data = torch.FloatTensor(dataset_array)
        
        # Dynamically build the save path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        save_path = os.path.join(project_root, "data", "processed", "lakh_tensor.pt")
        
        # Save it to your processed folder
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(tensor_data, save_path)
        print(f"\nData saved to {save_path}. You can now load this in your VAE training script!")
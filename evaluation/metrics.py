import pretty_midi
import numpy as np
import os
import random

def generate_random_baseline(num_samples=5, sequence_length=128):
    # Setup paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    out_dir = os.path.join(project_root, "outputs", "generated_midis", "baseline_random")
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Generating {num_samples} Random Baseline tracks...")
    
    for i in range(num_samples):
        midi = pretty_midi.PrettyMIDI()
        piano = pretty_midi.Instrument(program=0) # Acoustic Grand
        
        current_time = 0.0
        for _ in range(sequence_length):
            # Pick a completely random note between C3 (48) and C6 (84)
            pitch = random.randint(48, 84)
            
            # Random duration between 0.1s and 0.5s
            duration = random.uniform(0.1, 0.5)
            
            note = pretty_midi.Note(
                velocity=100, 
                pitch=pitch, 
                start=current_time, 
                end=current_time + duration
            )
            piano.notes.append(note)
            current_time += duration
            
        midi.instruments.append(piano)
        file_name = os.path.join(out_dir, f"random_baseline_{i+1}.mid")
        midi.write(file_name)
        print(f"Saved: {file_name}")

if __name__ == "__main__":
    generate_random_baseline()
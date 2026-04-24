import os
import random
import pretty_midi
import numpy as np

def generate_markov_baseline(num_samples=5, sequence_length=128):
    # Setup paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    out_dir = os.path.join(project_root, "outputs", "generated_midis", "baseline_markov")
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Generating {num_samples} Markov Chain baseline tracks...")
    
    # 1. Define a simple transition matrix (Probabilities of moving from one note to another)
    # For a real Markov chain, we would train this on the Lakh dataset. 
    # For a baseline, we simulate a C-Major scale bias.
    scale = [60, 62, 64, 65, 67, 69, 71, 72] # C4 to C5
    
    for i in range(num_samples):
        midi = pretty_midi.PrettyMIDI()
        piano = pretty_midi.Instrument(program=0)
        
        current_time = 0.0
        current_note = random.choice(scale)
        
        for _ in range(sequence_length):
            duration = random.uniform(0.2, 0.4)
            
            note = pretty_midi.Note(
                velocity=100, 
                pitch=current_note, 
                start=current_time, 
                end=current_time + duration
            )
            piano.notes.append(note)
            current_time += duration
            
            # Markov Transition: 70% chance to step to an adjacent note in the scale, 30% chance to jump
            if random.random() > 0.3:
                idx = scale.index(current_note)
                step = random.choice([-1, 1])
                new_idx = max(0, min(len(scale)-1, idx + step))
                current_note = scale[new_idx]
            else:
                current_note = random.choice(scale)
                
        midi.instruments.append(piano)
        file_name = os.path.join(out_dir, f"markov_baseline_{i+1}.mid")
        midi.write(file_name)
        print(f"Saved: {file_name}")

if __name__ == "__main__":
    generate_markov_baseline()
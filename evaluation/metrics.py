import pretty_midi
import numpy as np
import os
import random
import glob

# Import your new metric functions
from pitch_histogram import get_pitch_class_histogram
from rhythm_score import calculate_rhythm_diversity

def generate_random_baseline(num_samples=5, sequence_length=128):
    """Generates random baseline tracks."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    out_dir = os.path.join(project_root, "outputs", "generated_midis", "baseline_random")
    os.makedirs(out_dir, exist_ok=True)
    
    for i in range(num_samples):
        midi = pretty_midi.PrettyMIDI()
        piano = pretty_midi.Instrument(program=0)
        current_time = 0.0
        for _ in range(sequence_length):
            pitch = random.randint(48, 84)
            duration = random.uniform(0.1, 0.5)
            note = pretty_midi.Note(velocity=100, pitch=pitch, start=current_time, end=current_time + duration)
            piano.notes.append(note)
            current_time += duration
        midi.instruments.append(piano)
        midi.write(os.path.join(out_dir, f"random_baseline_{i+1}.mid"))

def evaluate_all_models():
    """Calculates metrics for all generated MIDIs and prints a comparison table."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    midi_dir = os.path.join(project_root, "outputs", "generated_midis")
    
    # We define the patterns to find your generated files
    models = {
        "Random Baseline": os.path.join(midi_dir, "baseline_random", "*.mid"),
        "Markov Baseline": os.path.join(midi_dir, "baseline_markov", "*.mid"),
        "Task 1 (LSTM)": os.path.join(midi_dir, "task1_lstm_*.mid"),
        "Task 2 (VAE)": os.path.join(midi_dir, "vae_generated_*.mid"),
        "Task 3 (Transformer)": os.path.join(midi_dir, "task3_transformer_*.mid"),
        "Task 4 (RLHF)": os.path.join(midi_dir, "task4_rlhf_*.mid"),
    }

    print("\n" + "="*60)
    print(f"{'Model':<25} | {'Avg Rhythm Diversity':<20}")
    print("-" * 60)

    for model_name, path_pattern in models.items():
        files = glob.glob(path_pattern)
        if not files:
            print(f"{model_name:<25} | No files found")
            continue
            
        rhythm_scores = []
        for f in files:
            score = calculate_rhythm_diversity(f)
            rhythm_scores.append(score)
            
        avg_rhythm = np.mean(rhythm_scores)
        print(f"{model_name:<25} | {avg_rhythm:.4f}")
        
    print("="*60 + "\n")

if __name__ == "__main__":
    # 1. Generate the random baseline first
    print("Ensuring random baselines exist...")
    generate_random_baseline()
    
    # 2. Run the evaluation
    print("Evaluating models...")
    evaluate_all_models()
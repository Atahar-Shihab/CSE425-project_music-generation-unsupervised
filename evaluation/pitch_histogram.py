import pretty_midi
import numpy as np
import os

def get_pitch_class_histogram(midi_file_path):
    """Calculates the normalized pitch class distribution (12 bins) for a MIDI file."""
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_file_path)
    except Exception as e:
        return np.zeros(12)
        
    pitch_counts = np.zeros(12)
    for instrument in midi_data.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                pitch_class = note.pitch % 12
                pitch_counts[pitch_class] += 1
                
    total_notes = np.sum(pitch_counts)
    if total_notes > 0:
        return pitch_counts / total_notes
    return pitch_counts

def calculate_pitch_histogram_similarity(midi_p_path, midi_q_path):
    """Calculates H(p,q) = sum(|p_i - q_i|) between two MIDI files."""
    p_hist = get_pitch_class_histogram(midi_p_path)
    q_hist = get_pitch_class_histogram(midi_q_path)
    return np.sum(np.abs(p_hist - q_hist))
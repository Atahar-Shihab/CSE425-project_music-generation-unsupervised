import pretty_midi

def calculate_rhythm_diversity(midi_file_path, tolerance=0.05):
    """Calculates the ratio of unique note durations to total notes."""
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_file_path)
    except Exception as e:
        return 0.0

    durations = []
    for instrument in midi_data.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                duration = round((note.end - note.start) / tolerance) * tolerance
                durations.append(duration)
    
    total_notes = len(durations)
    if total_notes == 0:
        return 0.0
        
    return len(set(durations)) / total_notes
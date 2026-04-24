import torch

class Config:
    # Path Configuration
    PROJECT_ROOT = "D:/music-generation-unsupervised"
    DATA_RAW = f"{PROJECT_ROOT}/data/raw_midi"
    DATA_PROCESSED = f"{PROJECT_ROOT}/data/processed"
    DATA_SPLITS = f"{PROJECT_ROOT}/data/train_test_split"
    OUTPUTS = f"{PROJECT_ROOT}/outputs"
    
    # Model Hyperparameters
    FEATURE_DIM = 88   # Standard Piano Keys
    SEQ_LEN = 128     # Number of time steps per sample
    LATENT_DIM = 128
    HIDDEN_DIM = 256
    
    # Training Hyperparameters
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    EPOCHS = 10
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # MIDI Configuration
    FS = 4            # Sampling frequency (ticks per beat)
    MIN_PITCH = 21    # A0
    MAX_PITCH = 108   # C8
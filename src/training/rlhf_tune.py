import torch
import torch.optim as optim
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from src.models.vae import MusicVAE

def simulated_human_reward(generated_music):
    """
    Simulates a human listening score [1, 5] based on how harmonic it is.
    Rewards notes that fall into the C-Major scale (white keys).
    """
    c_major_scale = [0, 2, 4, 5, 7, 9, 11] # C, D, E, F, G, A, B
    
    # Flatten and find which pitches were played
    # Fix: The sequence is 2D (Time, Pitch), so Pitch is at index 1
    active_pitches = torch.where(generated_music > 0.5)[1] 
    
    if len(active_pitches) == 0:
        return 1.0 # Score 1 for silence
        
    # Check what percentage of notes are in C Major
    in_scale_count = sum(1 for p in active_pitches if p.item() % 12 in c_major_scale)
    harmony_ratio = in_scale_count / len(active_pitches)
    
    # Map ratio to a 1-5 score
    score = 1.0 + (harmony_ratio * 4.0)
    return score

def rlhf_policy_gradient():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Starting RLHF Fine-tuning (Simulated Human Feedback)...")
    
    # Load the VAE we already trained
    model = MusicVAE().to(device)
    model_path = os.path.join(project_root, "outputs", "vae_model.pt")
    model.load_state_dict(torch.load(model_path, weights_only=True))
    
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    rl_steps = 50
    
    model.train()
    for step in range(rl_steps):
        optimizer.zero_grad()
        
        # 1. Generate music sample
        z = torch.randn(16, 128).to(device) # Batch of 16
        generated_music = model.decode(z).view(-1, 128, 88)
        
        # 2. Get simulated human reward score r = HumanScore(X_gen)
        rewards = torch.tensor([simulated_human_reward(seq) for seq in generated_music]).to(device)
        avg_reward = rewards.mean().item()
        
        # 3. Policy Gradient Update
        # J(theta) = E[r * log p_theta(X)]
        log_probs = torch.log(generated_music + 1e-8) # Add epsilon to prevent log(0)
        
        # Calculate loss weighted by the reward 
        loss = -torch.mean(log_probs * rewards.view(-1, 1, 1))
        
        loss.backward()
        optimizer.step()
        
        if (step+1) % 10 == 0:
            print(f"RL Step [{step+1}/{rl_steps}] \t Average Human Score: {avg_reward:.2f}/5.00")
            
    # Save tuned model
    torch.save(model.state_dict(), os.path.join(project_root, "outputs", "vae_rlhf_model.pt"))
    print("\nTask 4 Complete! RLHF-tuned weights saved.")

if __name__ == "__main__":
    rlhf_policy_gradient()
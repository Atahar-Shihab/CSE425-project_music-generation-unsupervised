import torch
import torch.optim as optim
import torch.nn.functional as F
import os
import sys
import copy

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from src.models.vae import MusicVAE

def simulated_human_reward(generated_music):
    c_major_scale = [0, 2, 4, 5, 7, 9, 11]
    active_pitches = torch.where(generated_music > 0.5)[1] 
    
    if len(active_pitches) == 0:
        return 0.0 
        
    in_scale_count = sum(1 for p in active_pitches if (p.item() + 21) % 12 in c_major_scale)
    harmony_ratio = in_scale_count / len(active_pitches)
    
    return 1.0 + (harmony_ratio * 3.6)

def train_rlhf():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Starting KL-Regularized RLHF (True Policy Gradient)...")
    
    # 1. Load the trainable model
    model = MusicVAE().to(device)
    model_path = os.path.join(project_root, "outputs", "vae_model.pt")
    model.load_state_dict(torch.load(model_path, weights_only=True))
    
    # 2. Create the FROZEN reference model to prevent Reward Hacking
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    rl_steps = 250
    beta_kl = 0.1 # The weight of the KL penalty
    
    model.train()
    for step in range(rl_steps):
        optimizer.zero_grad()
        
        z = torch.randn(64, 128).to(device) 
        
        # Get probabilities from both models
        active_probs = model.decode(z).view(-1, 128, 88)
        with torch.no_grad():
            ref_probs = ref_model.decode(z).view(-1, 128, 88)
        
        # Calculate Rewards
        with torch.no_grad():
            eval_music = (active_probs > 0.5).float()
            rewards = torch.tensor([simulated_human_reward(seq) for seq in eval_music]).to(device)
            # Advantage normalization to stabilize gradients
            advantages = rewards - rewards.mean()
            
        # Calculate Policy Gradient Loss (pushing up probabilities of high-reward sequences)
        log_probs = torch.log(active_probs + 1e-8)
        pg_loss = -torch.mean(log_probs * advantages.view(-1, 1, 1))
        
        # Calculate KL Divergence Penalty (keeping it close to the original music knowledge)
        kl_div = F.kl_div(log_probs, ref_probs, reduction='batchmean')
        
        # Total Loss
        loss = pg_loss + (beta_kl * kl_div)
        
        loss.backward()
        optimizer.step()
        
        if (step+1) % 25 == 0:
            print(f"Step [{step+1}/{rl_steps}] \t Average Human Score: {rewards.mean().item():.2f}/5.00 \t KL Penalty: {kl_div.item():.4f}")
            
    torch.save(model.state_dict(), os.path.join(project_root, "outputs", "vae_rlhf_model.pt"))
    print("\nTask 4 Complete! Academically rigorous RLHF weights updated.")

if __name__ == "__main__":
    train_rlhf()
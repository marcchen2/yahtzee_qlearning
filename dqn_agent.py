import torch
import torch.nn as nn
import torch.optim as optim

from collections import deque
import random
import numpy as np

class DQN(nn.Module):
    def __init__(self, state_dim=40, action_dim=45, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )
        
    def forward(self, x):
        return self.net(x)  # [batch_size, action_dim]


class ReplayBuffer:
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        # Check for GPU availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def push(self, state, action, reward, next_state, done, next_valid_mask):
        # Convert to CPU tensors if needed
        if isinstance(state, torch.Tensor):
            state = state.cpu()
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.cpu()
        self.buffer.append((state, action, reward, next_state, done, next_valid_mask))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, next_valid_masks = zip(*batch)
        
        # Convert to tensors and move to GPU
        return (
            torch.stack([s if isinstance(s, torch.Tensor) else torch.FloatTensor(s) for s in states]).to(self.device),
            torch.tensor(actions, dtype=torch.long).to(self.device),
            torch.tensor(rewards, dtype=torch.float32).to(self.device),
            torch.stack([ns if isinstance(ns, torch.Tensor) else torch.FloatTensor(ns) for ns in next_states]).to(self.device),
            torch.tensor(dones, dtype=torch.float32).to(self.device),
            torch.stack(next_valid_masks)
        )


# def select_action(dqn, state, epsilon):
#     """
#     - state: a single state tensor of shape (40,) (unbatched).
#     - epsilon: probability of random action.
#     - returns: integer in [0..44].
#     """
#     if random.random() < epsilon:
#         # Explore
#         return random.randint(0, 44)
#     else:
#         # Exploit
#         with torch.no_grad():
#             q_values = dqn(state.unsqueeze(0))  # shape (1, 45)
#             action = q_values.argmax(dim=1).item()
#         return action

def select_action(dqn, state, valid_actions_mask, epsilon):
    """
    - state: numpy array or tensor
    - valid_actions_mask: list of booleans
    """
    # Convert to tensor and move to DQN's device
    if not isinstance(state, torch.Tensor):
        state = torch.FloatTensor(state).to(next(dqn.parameters()).device)
    else:
        state = state.to(next(dqn.parameters()).device)
    
    valid_indices = [i for i, valid in enumerate(valid_actions_mask) if valid]
    
    if random.random() < epsilon:
        return random.choice(valid_indices)
    else:
        with torch.no_grad():
            q_values = dqn(state.unsqueeze(0))
            # Move mask to same device as DQN
            mask_tensor = torch.tensor(valid_actions_mask, 
                                     dtype=torch.bool).to(q_values.device)
            masked_q = q_values.clone()
            masked_q[0, ~mask_tensor] = -float('inf')
            action = masked_q.argmax().item()
            return action if action in valid_indices else random.choice(valid_indices)


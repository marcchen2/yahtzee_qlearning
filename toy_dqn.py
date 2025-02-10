import torch
import torch.nn as nn
import torch.optim as optim

from collections import deque
import random
import numpy as np

class DQN(nn.Module):
    def __init__(self, state_dim=9, action_dim=13, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )
        
    def forward(self, x):
        return self.net(x)  # shape: [batch_size, action_dim]

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def push(self, state, action, reward, next_state, done):
        """
        Store a transition. 
        Expecting state, next_state as CPU or float arrays if needed.
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to Tensors
        states_t = torch.stack([
            s if isinstance(s, torch.Tensor) else torch.FloatTensor(s)
            for s in states
        ])
        actions_t = torch.LongTensor(actions)
        rewards_t = torch.FloatTensor(rewards)
        next_states_t = torch.stack([
            ns if isinstance(ns, torch.Tensor) else torch.FloatTensor(ns)
            for ns in next_states
        ])
        dones_t = torch.FloatTensor(dones)

        return (
            states_t.to(self.device),
            actions_t.to(self.device),
            rewards_t.to(self.device),
            next_states_t.to(self.device),
            dones_t.to(self.device)
        )

def select_action(dqn, state, valid_actions_mask, epsilon):
    """
    Given a single state (tensor shape [9]) and a valid_actions_mask (length 13 booleans),
    select an action index in [0..12].
    """
    # Move state to correct device
    if not isinstance(state, torch.Tensor):
        state = torch.FloatTensor(state).to(next(dqn.parameters()).device)
    else:
        state = state.to(next(dqn.parameters()).device)

    # Indices of valid actions
    valid_indices = [i for i, valid in enumerate(valid_actions_mask) if valid]

    # Epsilon-greedy
    if random.random() < epsilon:
        return random.choice(valid_indices)
    else:
        with torch.no_grad():
            q_values = dqn(state.unsqueeze(0))  # shape [1, 13]
            # Mask out invalid actions
            mask = torch.BoolTensor(valid_actions_mask).to(q_values.device)
            q_values[0, ~mask] = -float('inf')
            action = q_values.argmax(dim=1).item()
        return action

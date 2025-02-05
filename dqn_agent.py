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
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (torch.stack(states), 
                torch.tensor(actions, dtype=torch.long),
                torch.tensor(rewards, dtype=torch.float32),
                torch.stack(next_states),
                torch.tensor(dones, dtype=torch.float32))
    
    def __len__(self):
        return len(self.buffer)


def select_action(dqn, state, epsilon):
    """
    - state: a single state tensor of shape (40,) (unbatched).
    - epsilon: probability of random action.
    - returns: integer in [0..44].
    """
    if random.random() < epsilon:
        # Explore
        return random.randint(0, 44)
    else:
        # Exploit
        with torch.no_grad():
            q_values = dqn(state.unsqueeze(0))  # shape (1, 45)
            action = q_values.argmax(dim=1).item()
        return action



import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import deque
import random
import numpy as np
from collections import namedtuple
import math 


class NoisyLinear(nn.Module):
    """
    Noisy linear layer for NoisyNets exploration (Fortunato et al.).
    """
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        # Learnable parameters for the “base” mu and sigma
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias_mu = nn.Parameter(torch.empty(out_features))
            self.bias_sigma = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_sigma', None)

        # Buffers to store the generated noise (not learnable, but saved to GPU/CPU device)
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        if bias:
            self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """
        Initialize (mu, sigma) parameters. Typically mu ~ Uniform and sigma is a small constant.
        """
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init * mu_range)
        if self.bias_mu is not None:
            self.bias_mu.data.uniform_(-mu_range, mu_range)
            self.bias_sigma.data.fill_(self.sigma_init * mu_range)

    def reset_noise(self):
        """
        Sample new noise matrices for weight and bias.
        Ensures noise is generated on the same device as the parameters.
        """
        device = self.weight_mu.device  # Ensure noise is on the same device

        eps_in = self._scale_noise(self.in_features).to(device)
        eps_out = self._scale_noise(self.out_features).to(device)

        # Outer product for factorized Gaussian noise
        self.weight_epsilon.copy_(eps_out.ger(eps_in))
        if self.bias_mu is not None:
            self.bias_epsilon.copy_(eps_out)

    def forward(self, x):
        if self.training:
            # Add noise only during training
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = None
            if self.bias_mu is not None:
                bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            # No noise during eval
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def _scale_noise(self, size):
        # Factorized noise per Fortunato et al.
        x = torch.randn(size, device=self.weight_mu.device)  # Ensure noise is on the correct device
        return x.sign() * x.abs().sqrt()
    
    

class DQN(nn.Module):
    def __init__(self, state_dim=40, action_dim=45, hidden=256):
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


# class DuelingDQN(nn.Module):
#     def __init__(self, state_dim, action_dim, hidden_dim=512):
#         super(DuelingDQN, self).__init__()
        
#         # Shared feature extraction
#         self.feature_layer = nn.Sequential(
#             nn.Linear(state_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU()
#         )
        
#         # Value stream (outputs a single value, V(s))
#         self.value_stream = nn.Linear(hidden_dim, 1)
        
#         # Advantage stream (outputs advantage for each action, A(s,a))
#         self.advantage_stream = nn.Linear(hidden_dim, action_dim)

#     def forward(self, x):
#         # Extract shared features
#         features = self.feature_layer(x)
        
#         # Compute the value and advantage
#         values = self.value_stream(features)              # shape: [batch_size, 1]
#         advantages = self.advantage_stream(features)      # shape: [batch_size, action_dim]
        
#         # Subtract mean advantage to keep Q-values stable
#         # Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
#         advantages_mean = advantages.mean(dim=1, keepdim=True)
#         Q = values + (advantages - advantages_mean)
        
#         return Q

class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=512):
        super(DuelingDQN, self).__init__()
        
        # Shared feature extraction using NoisyLinear
        self.feature_layer = nn.Sequential(
            NoisyLinear(state_dim, hidden_dim),
            nn.ReLU(),
            NoisyLinear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Value stream (outputs a single value, V(s))
        self.value_stream = NoisyLinear(hidden_dim, 1)
        
        # Advantage stream (outputs advantage for each action, A(s,a))
        self.advantage_stream = NoisyLinear(hidden_dim, action_dim)

    def forward(self, x):
        # Extract shared features
        features = self.feature_layer(x)
        
        # Compute the value and advantage
        values = self.value_stream(features)              # shape: [batch_size, 1]
        advantages = self.advantage_stream(features)      # shape: [batch_size, action_dim]
        
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,a)) across actions
        advantages_mean = advantages.mean(dim=1, keepdim=True)
        Q = values + (advantages - advantages_mean)
        
        return Q
    
    def reset_noise(self):
        """
        Resets the noise parameters in all noisy layers.
        Call this at each training step (or each time step) so
        that the agent explores differently on each decision.
        """
        # If your feature layer is a Sequential, you can index them directly:
        for module in self.feature_layer:
            if isinstance(module, NoisyLinear):
                module.reset_noise()
        
        self.value_stream.reset_noise()
        self.advantage_stream.reset_noise()



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


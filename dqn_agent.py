import torch
import torch.nn as nn
import torch.optim as optim

from collections import deque
import random
import numpy as np
from collections import namedtuple

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

Transition = namedtuple('Transition', 
                        ('state', 'action', 'reward', 'next_state', 'done', 'next_valid_mask'))

class PrioritizedReplayBuffer:
    """
    A naive implementation of a Prioritized Experience Replay buffer.
    """
    def __init__(self, capacity, alpha=0.6):
        """
        Args:
            capacity (int): Maximum size of the replay buffer.
            alpha (float): How much prioritization is used 
                           (0 = no prioritization, 1 = full prioritization).
        """
        self.capacity = capacity
        self.alpha = alpha
        
        self.buffer = []
        self.priorities = []
        self.position = 0
        self.max_priority = 1.0  # start all new transitions with max priority

    def push(self, state, action, reward, next_state, done, next_valid_mask):
        """
        Adds a transition into the buffer with the maximum priority so that 
        it will more likely be sampled soon.
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
            self.priorities.append(None)

        transition = Transition(state.cpu(), action, reward, next_state.cpu(), done, next_valid_mask)
        
        self.buffer[self.position] = transition
        # Use the max priority so that a new experience is likely to be sampled
        self.priorities[self.position] = self.max_priority

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        """
        Samples a batch of experiences, returning transitions, indices, and IS weights.

        beta: controls to what degree we apply importance sampling correction
              (0 = no corrections, 1 = full correction).
        """
        # Calculate the probabilities for each entry
        # p_i = priorities_i ^ alpha
        # P(i) = p_i / sum_j p_j
        scaled_priorities = np.array(self.priorities)**self.alpha
        probs = scaled_priorities / np.sum(scaled_priorities)
        
        # Sample indices according to their probabilities
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        
        # Compute importance-sampling weights
        # w_i = (1 / (N * P(i)))^beta
        # normalized by max weight so that they scale to [0, 1]
        total = len(self.buffer)
        weights = (total * probs[indices])**(-beta)
        weights = weights / weights.max()  # normalize for stability
        
        # Now collect transitions
        transitions = [self.buffer[idx] for idx in indices]
        
        # Convert them to torch tensors
        states = torch.stack([t.state if isinstance(t.state, torch.Tensor) 
                              else torch.FloatTensor(t.state) for t in transitions])
        actions = torch.LongTensor([t.action for t in transitions])
        rewards = torch.FloatTensor([t.reward for t in transitions])
        next_states = torch.stack([t.next_state if isinstance(t.next_state, torch.Tensor) 
                                   else torch.FloatTensor(t.next_state) for t in transitions])
        dones = torch.FloatTensor([t.done for t in transitions])
        
        # next_valid_masks is a list of Tensors or lists, stack them
        next_valid_masks = torch.stack([t.next_valid_mask for t in transitions])

        # Return everything along with the indices and weights
        return (states, actions, rewards, next_states, dones, next_valid_masks), indices, torch.FloatTensor(weights)

    def update_priorities(self, indices, priorities):
        """
        Update the priorities of the sampled transitions based on new TD-error.
        """
        for idx, priority in zip(indices, priorities):
            # plus a small epsilon to avoid zero priority
            self.priorities[idx] = float(priority)
            # Update max priority
            self.max_priority = max(self.max_priority, float(priority))


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


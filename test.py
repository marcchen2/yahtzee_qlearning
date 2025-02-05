import random
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

###############################
# Toy Environment Code
###############################

def toy_encode_state(game_state):
    """
    Example minimal encoder for:
      - 3 dice
      - 3 categories: "Ones", "Twos", "Chance"
      - up to 2 rolls each turn (rolls_left in {0,1,2})
    
    Proposed 16 dims:
      0-5:   dice_counts (#1..#6)
      6-8:   sorted_dice / 6.0
      9:     rolls_left / 2.0  (normalized to [0..1])
      10-12: category availability
      13-15: category scores (normalized)
    """
    dice = sorted(game_state['dice'])
    dice_counts = torch.zeros(6)
    for d in game_state['dice']:
        dice_counts[d - 1] += 1
    
    # normalized dice
    sorted_dice = torch.tensor(dice, dtype=torch.float32) / 6.0
    
    # rolls_left / 2 => in [0,1]
    rolls_left = torch.tensor([game_state['rolls_left'] / 2.0], dtype=torch.float32)
    
    # 3 categories
    categories = ['Ones', 'Twos', 'Chance']
    cat_avail = torch.zeros(3)
    cat_score = torch.zeros(3)
    for i, cat in enumerate(categories):
        score_val = game_state['categories'][cat]
        if score_val is None:
            cat_avail[i] = 1.0
        else:
            # e.g. simple normalization by 15
            cat_score[i] = score_val / 15.0
    
    return torch.cat([dice_counts, sorted_dice, rolls_left, cat_avail, cat_score])

class MiniYahtzeeGame:
    """
    A toy environment with:
      - 3 dice
      - 3 categories: "Ones", "Twos", "Chance"
      - up to 2 rolls per turn
      - ends after all 3 categories are used
    """
    def __init__(self):
        self.categories = OrderedDict([
            ('Ones', None),
            ('Twos', None),
            ('Chance', None)
        ])
        self.dice = [0,0,0]
        self.rolls_left = 0  # increment to 2 when a new turn starts

    def reset(self):
        for cat in self.categories:
            self.categories[cat] = None
        self.dice = [0,0,0]
        self.rolls_left = 0
        return self.get_state()

    def get_state(self):
        return {
            'categories': self.categories.copy(),
            'dice': self.dice.copy(),
            'rolls_left': self.rolls_left
        }

    def get_encoded_state(self):
        return toy_encode_state(self.get_state())

    def calculate_score(self, category, dice):
        if category == 'Ones':
            return sum(d for d in dice if d == 1)
        elif category == 'Twos':
            return sum(d for d in dice if d == 2)
        elif category == 'Chance':
            return sum(dice)
        return 0

    def roll_dice(self, keep_mask=None):
        if keep_mask is None:
            # roll all dice
            self.dice = [random.randint(1,6) for _ in range(3)]
        else:
            for i in range(3):
                if not keep_mask[i]:
                    self.dice[i] = random.randint(1,6)
        self.dice.sort()

    def step(self, action):
        """
        action: ('reroll', keep_mask) or ('score', category)
        """
        reward = 0.0
        done = False

        if action[0] == 'reroll':
            keep_mask = action[1]
            # If starting a new turn, set rolls_left=2
            if self.rolls_left == 0:
                self.rolls_left = 2
            if self.rolls_left > 0:
                self.roll_dice(keep_mask)
                self.rolls_left -= 1

        elif action[0] == 'score':
            category = action[1]
            if self.categories[category] is None:
                # Calculate and assign score
                sc = self.calculate_score(category, self.dice)
                self.categories[category] = sc
                reward = sc
            # End the turn
            self.rolls_left = 0

            # Check if all categories are filled
            if all(v is not None for v in self.categories.values()):
                done = True

        next_state = self.get_state()
        return next_state, reward, done, {}

###############################
# Action Space
###############################

def toy_generate_all_actions():
    """
    e.g. 8 keep masks for 3 dice (2^3) + 1 "roll all" (None) + 3 score categories
    = 8 + 1 + 3 = 12 total
    """
    actions = []
    
    # reroll keep masks
    for mask_int in range(8):  # 2^3
        keep_mask = [(mask_int & (1 << i)) != 0 for i in range(3)]
        actions.append(('reroll', keep_mask))
    # Also a "reroll-all" action (None)
    actions.append(('reroll', None))

    # 3 scoring categories
    for cat in ['Ones', 'Twos', 'Chance']:
        actions.append(('score', cat))

    return actions

ALL_ACTIONS = toy_generate_all_actions()

###############################
# DQN Components
###############################

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = zip(*batch)
        # s, s2 are tensors
        return (torch.stack(s),
                torch.tensor(a, dtype=torch.long),
                torch.tensor(r, dtype=torch.float32),
                torch.stack(s2),
                torch.tensor(d, dtype=torch.float32))
    
    def __len__(self):
        return len(self.buffer)

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    def forward(self, x):
        return self.net(x)

def select_action(dqn, state, epsilon):
    """
    - state: (16,) unbatched
    - returns an int in [0..(action_dim-1)]
    """
    if random.random() < epsilon:
        return random.randint(0, dqn.forward(state.unsqueeze(0)).shape[1]-1)
    else:
        with torch.no_grad():
            q_values = dqn(state.unsqueeze(0))  # shape (1, action_dim)
            action = q_values.argmax(dim=1).item()
        return action

###############################
# Training Loop
###############################

def train_toy_dqn(env_cls, 
                  ALL_ACTIONS,
                  num_episodes=1000,
                  buffer_capacity=1000,
                  batch_size=32,
                  gamma=0.99,
                  lr=1e-3,
                  epsilon_start=1.0,
                  epsilon_end=0.1,
                  epsilon_decay=50000,
                  update_target_every=500):
    """
    Trains a DQN on the toy environment.
    """
    # Infer dimensions
    state_dim = 16   # from toy_encode_state
    action_dim = len(ALL_ACTIONS)  # 12
    
    # Create DQN, target net, replay buffer
    dqn = DQN(state_dim=state_dim, action_dim=action_dim)
    target_dqn = DQN(state_dim=state_dim, action_dim=action_dim)
    target_dqn.load_state_dict(dqn.state_dict())
    target_dqn.eval()  # Target net is in eval mode
    
    optimizer = optim.Adam(dqn.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(buffer_capacity)
    
    # Epsilon schedule
    def get_epsilon(step):
        return max(epsilon_end, epsilon_start - (step / epsilon_decay)*(epsilon_start - epsilon_end))
    
    total_steps = 0
    
    for episode in range(1, num_episodes + 1):
        env = env_cls()
        env.reset()
        state = env.get_encoded_state()
        done = False
    
        while not done:
            eps = get_epsilon(total_steps)
            action_idx = select_action(dqn, state, eps)
            action = ALL_ACTIONS[action_idx]
    
            next_state, reward, done, _ = env.step(action)
            next_state_encoded = next_state.get_encoded_state()
    
            replay_buffer.push(state, action_idx, reward, next_state_encoded, done)
            state = next_state_encoded
            total_steps += 1
    
            # Train step
            if len(replay_buffer) >= batch_size:
                s_batch, a_batch, r_batch, s2_batch, d_batch = replay_buffer.sample(batch_size)
    
                q_values = dqn(s_batch)
                q_selected = q_values.gather(1, a_batch.view(-1,1)).squeeze(1)
    
                with torch.no_grad():
                    q_next = target_dqn(s2_batch)
                    max_q_next = q_next.max(dim=1)[0]
    
                q_target = r_batch + gamma * max_q_next * (1 - d_batch)
    
                loss = nn.MSELoss()(q_selected, q_target)
    
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
            # Update target network periodically
            if total_steps % update_target_every == 0:
                target_dqn.load_state_dict(dqn.state_dict())
    
        # End of an episode -> optionally print progress
        if episode % 100 == 0:
            # final score
            final_score = sum(v for v in env.categories.values() if v is not None)
            print(f"Episode {episode}: final_score={final_score}, epsilon={eps:.3f}")
    
    return dqn, target_dqn

###############################
# Evaluation Function
###############################

def evaluate_agent(dqn, env_cls, ALL_ACTIONS, n_eval_episodes=100, max_steps_per_episode=100):
    """
    Evaluate the trained DQN agent over n_eval_episodes with epsilon=0.
    Returns the average and median final score.
    
    Args:
        dqn: Trained DQN model.
        env_cls: Callable that returns a new environment instance.
        ALL_ACTIONS: List of all possible actions.
        n_eval_episodes: Number of episodes to evaluate.
        max_steps_per_episode: Maximum steps to prevent infinite loops.
    
    Returns:
        avg_score: Average final score over all episodes.
        median_score: Median final score over all episodes.
    """
    dqn.eval()  # Set DQN to evaluation mode
    scores = []
    
    for episode in range(1, n_eval_episodes + 1):
        env = env_cls()
        state = env.reset()
        done = False
        steps = 0
        actions_taken = []
        
        while not done and steps < max_steps_per_episode:
            with torch.no_grad():
                # Greedy action selection (epsilon=0)
                q_values = dqn(state.unsqueeze(0))  # shape (1, action_dim)
                action_idx = q_values.argmax(dim=1).item()
            
            action = ALL_ACTIONS[action_idx]
            actions_taken.append(action)
            next_state, reward, done, _ = env.step(action)
            
            # **CORRECTION HERE**:
            # Call get_encoded_state() on the env instance, not on next_state (which is a dict)
            state = env.get_encoded_state()
            steps += 1
        
        # Check if the episode terminated normally or was cut off
        if steps >= max_steps_per_episode and not done:
            print(f"Episode {episode}: Reached max steps without completing the game.")
            print(f"Actions taken: {actions_taken}")
        
        # Compute final score
        if hasattr(env, 'upper_bonus'):
            final_score = sum(v for v in env.categories.values() if v is not None)
            final_score += env.upper_bonus + env.yahtzee_bonuses
        else:
            final_score = sum(v for v in env.categories.values() if v is not None)
        
        scores.append(final_score)
        
        # Optional: Log every N episodes
        if episode % 10 == 0:
            print(f"Episode {episode}/{n_eval_episodes} | Final Score: {final_score} | Steps: {steps}")
    
    avg_score = np.mean(scores)
    median_score = np.median(scores)
    print(f"\nEvaluation over {n_eval_episodes} episodes:")
    print(f"Average Score: {avg_score:.2f}")
    print(f"Median Score: {median_score:.2f}")
    return avg_score, median_score



if __name__ == "__main__":
    # Initialize random seeds for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    
    # Train the DQN agent on the toy environment
    dqn, target_dqn = train_toy_dqn(
        env_cls=MiniYahtzeeGame,
        ALL_ACTIONS=ALL_ACTIONS,
        num_episodes=2000,
        buffer_capacity=1000,
        batch_size=32,
        gamma=0.95,
        lr=1e-3,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=20000,
        update_target_every=200
    )
    
    # Evaluate the trained agent
    avg_score, median_score = evaluate_agent(
        dqn=dqn,
        env_cls=MiniYahtzeeGame,
        ALL_ACTIONS=ALL_ACTIONS,
        n_eval_episodes=100,
        max_steps_per_episode=10  # Lower for toy environment
    )
    
    print(f"\nFinal Evaluation -> Average Score: {avg_score:.2f}, Median Score: {median_score:.2f}")

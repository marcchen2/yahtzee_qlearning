import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random

###############################
# Minimal DQN Agent Structures
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
# Main Training Loop
###############################

def train_toy_dqn(env_cls, 
                  ALL_ACTIONS,
                  num_episodes=1000,
                  buffer_capacity=500,
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
    action_dim = len(ALL_ACTIONS)  # ~12

    # Create DQN, target net, replay buffer
    dqn = DQN(state_dim=state_dim, action_dim=action_dim)
    target_dqn = DQN(state_dim=state_dim, action_dim=action_dim)
    target_dqn.load_state_dict(dqn.state_dict())

    optimizer = optim.Adam(dqn.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(buffer_capacity)

    # Epsilon schedule
    def get_epsilon(step):
        return max(epsilon_end, epsilon_start - (step / epsilon_decay)*(epsilon_start - epsilon_end))

    total_steps = 0

    for episode in range(num_episodes):
        env = env_cls()
        env.reset()
        state = env.get_encoded_state()
        done = False

        while not done:
            eps = get_epsilon(total_steps)
            action_idx = select_action(dqn, state, eps)
            action = ALL_ACTIONS[action_idx]

            next_state_dict, reward, done, _ = env.step(action)
            next_state = env.get_encoded_state()

            replay_buffer.push(state, action_idx, reward, next_state, done)
            state = next_state
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

            if total_steps % update_target_every == 0:
                target_dqn.load_state_dict(dqn.state_dict())

        # End of an episode -> optionally print progress
        if (episode+1) % 100 == 0:
            # final score
            final_score = sum(v for v in env.categories.values() if v is not None)
            print(f"Episode {episode+1}: final_score={final_score}, epsilon={eps:.3f}")

    return dqn, target_dqn


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
        env.reset()  # Correctly reset the environment
        state = env.get_encoded_state()  # Get encoded initial state
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
            next_state_dict, reward, done, _ = env.step(action)
            
            # Update state with the new encoded state from the environment
            state = env.get_encoded_state()
            steps += 1
        
        # # Check if the episode terminated normally or was cut off
        # if steps >= max_steps_per_episode and not done:
        #     print(f"Episode {episode}: Reached max steps without completing the game.")
        #     print(f"Actions taken: {actions_taken}")
        
        # Compute final score using the environment's attributes
        final_score = sum(v for v in env.categories.values() if v is not None)
        if hasattr(env, 'upper_bonus'):
            final_score += env.upper_bonus
        if hasattr(env, 'yahtzee_bonuses'):
            final_score += env.yahtzee_bonuses
        
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
    from toy_yahtzee import MiniYahtzeeGame, toy_generate_all_actions

    # Prepare the action list
    TOY_ACTIONS = toy_generate_all_actions()

    # Train
    dqn, target_dqn = train_toy_dqn(
        env_cls=MiniYahtzeeGame,
        ALL_ACTIONS=TOY_ACTIONS,
        num_episodes=200,
        buffer_capacity=1000,
        batch_size=64,
        gamma=0.95,
        lr=1e-3,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=20000,
        update_target_every=200
    )
    
    # # Now evaluate (greedy actions, no random exploration)
    avg_score, median_score = evaluate_agent(dqn, MiniYahtzeeGame, TOY_ACTIONS, n_eval_episodes=100)
    # print(f"Evaluation over 100 episodes -> Average score: {avg_score:.1f}, Median score: {median_score:.1f}")

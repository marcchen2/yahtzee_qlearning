import utils
import dqn_agent 
import torch.optim as optim
import torch
import torch.nn as nn
from yahtzee import YahtzeeGame
import numpy as np
import random 

ALL_ACTIONS = utils.generate_all_actions()
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

def train_dqn(env_cls, num_episodes=1000,
              buffer_capacity=500,
              batch_size=64,
              gamma=0.99,
              lr=1e-3,
              epsilon_start=1.0,
              epsilon_end=0.1,
              epsilon_decay=1000000,
              update_target_every=500):
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    
    # Create DQN and target net with DataParallel if multiple GPUs
    dqn = dqn_agent.DQN(state_dim=40, action_dim=len(ALL_ACTIONS)).to(device)
    if num_gpus > 1:
        dqn = nn.DataParallel(dqn)
    
    target_dqn = dqn_agent.DQN(state_dim=40, action_dim=len(ALL_ACTIONS)).to(device)
    if num_gpus > 1:
        target_dqn = nn.DataParallel(target_dqn)
    target_dqn.load_state_dict(dqn.state_dict())
    
    optimizer = optim.Adam(dqn.parameters(), lr=lr)
    replay_buffer = dqn_agent.ReplayBuffer(buffer_capacity)

    def get_epsilon(step):
        return max(epsilon_end, epsilon_start - (step / epsilon_decay)*(epsilon_start - epsilon_end))

    total_steps = 0

    for episode in range(num_episodes):
        env = env_cls()
        state_dict = env.reset()
        state = env.get_encoded_state()  # Assume this returns a numpy array
        
        done = False
        while not done:
            # Convert state to tensor and move to device
            state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
            eps = get_epsilon(total_steps)
            action_idx = dqn_agent.select_action(dqn, state_tensor, eps)
            action = ALL_ACTIONS[action_idx]

            next_state_dict, reward, done, _info = env.step(action)
            next_state = env.get_encoded_state()

            replay_buffer.push(state, action_idx, reward, next_state, done)
            state = next_state
            total_steps += 1

            if len(replay_buffer) >= batch_size:
                s_batch, a_batch, r_batch, s2_batch, d_batch = replay_buffer.sample(batch_size)
                # Move batches to device
                s_batch = s_batch.to(device)
                a_batch = a_batch.to(device)
                r_batch = r_batch.to(device)
                s2_batch = s2_batch.to(device)
                d_batch = d_batch.to(device)

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

        final_score = sum(v for v in env.categories.values() if v is not None)
        final_score += env.upper_bonus + env.yahtzee_bonuses
        if (episode+1) % 100 == 0:
            print(f"Episode {episode+1}: final_score = {final_score:.1f}, epsilon = {eps:.3f}")

    return dqn, target_dqn

def evaluate_agent(dqn, env_cls, ALL_ACTIONS, n_eval_episodes=100, max_steps_per_episode=50):
    device = next(dqn.parameters()).device  # Get device from model
    
    dqn.eval()
    scores = []
    
    for episode in range(n_eval_episodes):
        env = env_cls()
        env.reset()
        state = env.get_encoded_state()  # numpy array
        state = torch.tensor(state, dtype=torch.float32).to(device)
        
        done = False
        steps = 0
        
        while not done and steps < max_steps_per_episode:
            with torch.no_grad():
                q_values = dqn(state.unsqueeze(0))
                action_idx = q_values.argmax().item()
            
            action = ALL_ACTIONS[action_idx]
            _, reward, done, _ = env.step(action)
            state = env.get_encoded_state()
            state = state.float().to(device)
            steps += 1
        
        final_score = sum(v for v in env.categories.values() if v is not None)
        final_score += env.upper_bonus + env.yahtzee_bonuses
        scores.append(final_score)
        
        if (episode+1) % 10 == 0:
            print(f"Episode {episode+1}/{n_eval_episodes} | Score: {final_score}")

    avg_score = np.mean(scores)
    median_score = np.median(scores)
    print(f"\nEvaluation Results ({n_eval_episodes} episodes):")
    print(f"Average Score: {avg_score:.1f}")
    print(f"Median Score: {median_score:.1f}")
    return avg_score, median_score

if __name__ == "__main__":
    def make_env():
        return YahtzeeGame()

    dqn, target_dqn = train_dqn(make_env, num_episodes=10000)
    evaluate_agent(dqn.module if hasattr(dqn, 'module') else dqn, make_env, ALL_ACTIONS)
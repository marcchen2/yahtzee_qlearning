import utils
import dqn_agent 
import torch.optim as optim
import torch
import torch.nn as nn
from yahtzee import YahtzeeGame
import numpy as np
import random 

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ALL_ACTIONS = utils.generate_all_actions()
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

def evaluate_random_baseline(env_cls, num_episodes=100):
    """Evaluate random agent's performance over multiple episodes."""
    total_scores = 0.0
    
    for _ in range(num_episodes):
        env = env_cls()
        env.reset()
        done = False
        
        while not done:
            valid_actions_mask = env.get_valid_actions_mask()
            # Convert mask to list of valid action indices
            valid_actions = [i for i, valid in enumerate(valid_actions_mask) if valid]
            action_idx = np.random.choice(valid_actions)
            
            # Convert action index to game action (same as in DQN)
            if action_idx < 32:
                keep_mask = [bool(int(bit)) for bit in f"{action_idx:05b}"]
                action = ('reroll', keep_mask)
            else:
                cat_idx = action_idx - 32
                category = list(env.categories.keys())[cat_idx]
                action = ('score', category)
            
            # Execute action
            _, _, done, _ = env.step(action)
        
        # Calculate final score
        final_score = sum(v for v in env.categories.values() if v is not None)
        final_score += env.upper_bonus + env.yahtzee_bonuses
        total_scores += final_score
    
    avg_score = total_scores / num_episodes
    return avg_score

def evaluate_model(model, env_cls, num_episodes=100):
    """Evaluate the model's performance over multiple episodes without exploration."""
    model.eval()
    total_scores = 0.0
    
    with torch.no_grad():
        for _ in range(num_episodes):
            env = env_cls()
            state = torch.FloatTensor(env.reset()).to(device)
            done = False
            
            while not done:
                valid_actions_mask = env.get_valid_actions_mask()
                action_idx = dqn_agent.select_action(model, state, valid_actions_mask, epsilon=0.0)
                
                # Convert action index to game action
                if action_idx < 32:
                    keep_mask = [bool(int(bit)) for bit in f"{action_idx:05b}"]
                    action = ('reroll', keep_mask)
                else:
                    cat_idx = action_idx - 32
                    category = list(env.categories.keys())[cat_idx]
                    action = ('score', category)
                
                # Execute action
                next_state, _, done, _ = env.step(action)
                state = torch.FloatTensor(next_state).to(device)
            
            # Calculate final score
            final_score = sum(v for v in env.categories.values() if v is not None)
            final_score += env.upper_bonus + env.yahtzee_bonuses
            total_scores += final_score
    
    avg_score = total_scores / num_episodes
    model.train()
    return avg_score

def train_dqn(env_cls, num_episodes=10000,
              buffer_capacity=1000,
              batch_size=256,
              gamma=0.99,
              lr=1e-3,
              epsilon_start=1.0,
              epsilon_end=0.1,
              epsilon_decay=100000,
              update_target_every=100,
              eval_interval=1000,
              eval_episodes=100):
    """
    Train DQN agent with periodic evaluation against random baseline.
    """
    # Initialize networks
    dqn = dqn_agent.DQN(state_dim=40, action_dim=len(ALL_ACTIONS)).to(device)
    target_dqn = dqn_agent.DQN(state_dim=40, action_dim=len(ALL_ACTIONS)).to(device)
    target_dqn.load_state_dict(dqn.state_dict())
    optimizer = optim.Adam(dqn.parameters(), lr=lr)
    replay_buffer = dqn_agent.ReplayBuffer(buffer_capacity)

    def get_epsilon(step):
        return max(epsilon_end, epsilon_start - (step / epsilon_decay)*(epsilon_start - epsilon_end))

    total_steps = 0

    # Initial random baseline evaluation
    random_score = evaluate_random_baseline(env_cls, eval_episodes)
    print(f"Initial random baseline: {random_score:.1f}")

    for episode in range(num_episodes):
        env = env_cls()
        state = torch.FloatTensor(env.reset()).to(device)
        valid_actions_mask = env.get_valid_actions_mask()
        done = False

        while not done:
            epsilon = get_epsilon(total_steps)
            action_idx = dqn_agent.select_action(dqn, state, valid_actions_mask, epsilon)
            
            # Convert action index to game action
            if action_idx < 32:
                keep_mask = [bool(int(bit)) for bit in f"{action_idx:05b}"]
                action = ('reroll', keep_mask)
            else:
                cat_idx = action_idx - 32
                category = list(env.categories.keys())[cat_idx]
                action = ('score', category)
            
            # Execute action
            next_state, reward, done, _ = env.step(action)
            next_state = torch.FloatTensor(next_state).to(device)
            next_valid_mask = env.get_valid_actions_mask()
            
            # Store experience in replay buffer
            replay_buffer.push(state.cpu(), action_idx, reward, next_state.cpu(), done)
            
            state = next_state
            valid_actions_mask = next_valid_mask
            total_steps += 1

            # Training step
            if len(replay_buffer.buffer) >= batch_size:
                s_batch, a_batch, r_batch, s2_batch, d_batch = replay_buffer.sample(batch_size)
                
                # Convert to GPU tensors
                s_batch = s_batch.to(device)
                a_batch = a_batch.to(device)
                r_batch = r_batch.to(device)
                s2_batch = s2_batch.to(device)
                d_batch = d_batch.to(device)

                # Compute Q-values and loss
                q_values = dqn(s_batch)
                q_selected = q_values.gather(1, a_batch.unsqueeze(1)).squeeze(1)
                
                with torch.no_grad():
                    q_next = target_dqn(s2_batch)
                    max_q_next = q_next.max(dim=1)[0]
                
                q_target = r_batch + gamma * max_q_next * (1 - d_batch)
                loss = nn.MSELoss()(q_selected, q_target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Update target network
            if total_steps % update_target_every == 0:
                target_dqn.load_state_dict(dqn.state_dict())

        # Periodic evaluation
        if (episode + 1) % eval_interval == 0:
            model_score = evaluate_model(dqn, env_cls, eval_episodes)
            random_score = evaluate_random_baseline(env_cls, eval_episodes)
            print(f"\nEvaluation after episode {episode+1}:")
            print(f"DQN Average: {model_score:.1f}")
            print(f"Random Baseline Average: {random_score:.1f}\n")

        # Episode statistics
        if (episode + 1) % 100 == 0:
            final_score = sum(v for v in env.categories.values() if v is not None)
            final_score += env.upper_bonus + env.yahtzee_bonuses
            print(f"Episode {episode+1}: Training score = {final_score:.1f}, Epsilon = {epsilon:.3f}")

    return dqn, target_dqn

if __name__ == "__main__":
    def make_env():
        return YahtzeeGame()

    # dqn, target_dqn = train_dqn(make_env, 
    #                            num_episodes=10000,
    #                            eval_interval=1000,
    #                            eval_episodes=100)
    
    # env = YahtzeeGame()
    # env.reset()
    
    # done = False

    # while not done:
    #     print("Current dice:", env.dice)
    #     print("Available categories:", [k for k, v in env.categories.items() if v is None])
    #     action = input("Enter action (e.g., 'reroll 11000' or 'score three_of_a_kind'): ")
    #     next_state, reward, done, _ = env.step(action)
    
    # Example: Score all 1s in upper section
    env = YahtzeeGame()
    env.reset()
    
    env.categories["ones"] = 5  # Pretend perfect score
    env.categories["twos"] = 10  # Etc...
    print(env.calculate_final_score())  # Should be 63 (upper) + 35 (bonus) + ... 
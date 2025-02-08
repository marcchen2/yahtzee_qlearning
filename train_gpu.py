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

def train_dqn(env_cls, num_episodes=10000,
              buffer_capacity=1000,
              batch_size=256,
              gamma=0.99,
              lr=1e-3,
              epsilon_start=1.0,
              epsilon_end=0.1,
              epsilon_decay=100000,
              update_target_every=100):
    """
    Train DQN agent with GPU support.
    """
    # Initialize DQN and target DQN on the correct device
    dqn = dqn_agent.DQN(state_dim=40, action_dim=len(ALL_ACTIONS)).to(device)
    target_dqn = dqn_agent.DQN(state_dim=40, action_dim=len(ALL_ACTIONS)).to(device)
    target_dqn.load_state_dict(dqn.state_dict())
    optimizer = optim.Adam(dqn.parameters(), lr=lr)
    replay_buffer = dqn_agent.ReplayBuffer(buffer_capacity)

    def get_epsilon(step):
        return max(epsilon_end, epsilon_start - (step / epsilon_decay)*(epsilon_start - epsilon_end))

    
    total_steps = 0

    for episode in range(num_episodes):
        env = env_cls()
        state_dict = env.reset()
        state = env.get_encoded_state()  # CPU tensor
        valid_actions_mask = env.get_valid_actions_mask()
        
        # Convert to GPU tensor once here
        state = torch.FloatTensor(state).to(device)  

        done = False
        while not done:
            
            epsilon = get_epsilon(total_steps)
            action_idx = dqn_agent.select_action(dqn, state, valid_actions_mask, epsilon)
           
            # Convert action index to game action
            if action_idx < 32:
                # Reroll action: convert index to keep_mask (e.g., 5-bit binary)
                keep_mask = [bool(int(bit)) for bit in f"{action_idx:05b}"]
                action = ('reroll', keep_mask)
            else:
                # Scoring action: map to category name
                cat_idx = action_idx - 32
                category = list(env.categories.keys())[cat_idx]
                action = ('score', category)
            
            # Execute action
            next_state, reward, done, _ = env.step(action)
            next_valid_mask = env.get_valid_actions_mask()
            
            # Store experience in replay buffer
            replay_buffer.push(state, action_idx, reward, next_state, done)
            
            state = next_state
            valid_actions_mask = next_valid_mask
            
            # action_idx = dqn_agent.select_action(dqn, state.to(device), eps)
            # action = ALL_ACTIONS[action_idx]

            # next_state_dict, reward, done, _info = env.step(action)
            # next_state = env.get_encoded_state()  # CPU tensor

            # replay_buffer.push(state, action_idx, reward, next_state, done)
            # state = next_state  # Keep state on CPU for replay buffer
            total_steps += 1

            if len(replay_buffer.buffer) >= batch_size:
                # Sample and move batches to GPU
                s_batch, a_batch, r_batch, s2_batch, d_batch = replay_buffer.sample(batch_size)

                # Compute Q-values and loss on GPU
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
            def make_env():
                return YahtzeeGame()
            print(f"Episode {episode+1}: final_score = {final_score:.1f}, epsilon = {epsilon:.3f}")            

    return dqn, target_dqn


if __name__ == "__main__":
    def make_env():
        return YahtzeeGame()

    dqn, target_dqn = train_dqn(make_env, num_episodes=10000)
    # Evaluate the trained agent
    # evaluate_agent(dqn, make_env, ALL_ACTIONS, n_eval_episodes=1000)


#next : make better evaluation function
# track log loss
# then research about how to improve things 



import utils
import dqn_agent 
import torch.optim as optim
import torch
import torch.nn as nn
from yahtzee import YahtzeeGame
import numpy as np
import random 
import wandb

from torchrl.data import ListStorage, PrioritizedReplayBuffer

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ALL_ACTIONS = utils.generate_all_actions()
torch.manual_seed(1)
np.random.seed(1)
random.seed(1)

def evaluate_model(model, env_cls, num_episodes=100, epsilon=0):
    """Evaluate the model's performance over multiple episodes without exploration."""
    model.eval()  # Set model to evaluation mode
    total_scores = 0.0
    
    with torch.no_grad():  # Disable gradient computation
        for _ in range(num_episodes):
            env = env_cls()
            state_dict = env.reset()
            state = env.get_encoded_state()
            state = torch.FloatTensor(state).to(device)
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
                next_state, reward, done, _ = env.step(action)
                state = torch.FloatTensor(next_state).to(device)
            
            # Calculate final score
            final_score = sum(v for v in env.categories.values() if v is not None)
            final_score += env.upper_bonus + env.yahtzee_bonuses
            total_scores += final_score
    
    avg_score = total_scores / num_episodes
    model.train()  # Set model back to training mode
    return avg_score

def train_dqn(env_cls, num_episodes=10000,
              buffer_capacity=2000,
              batch_size=256,
              gamma=0.99,
              lr=1e-5,
              epsilon_start=1.0,
              epsilon_end=0.1,
              epsilon_decay_prop=0.7,  # % of total steps
              update_target_every=500,
              eval_interval=100,
              eval_episodes=100,
              max_grad_norm = 0.5):
    """
    Train DQN agent with periodic evaluation.
    """
    # Initialize networks
    input_length = len(env_cls().get_encoded_state())
    dqn = dqn_agent.DQN(state_dim=input_length, action_dim=len(ALL_ACTIONS)).to(device)
    target_dqn = dqn_agent.DQN(state_dim=input_length, action_dim=len(ALL_ACTIONS)).to(device)
    target_dqn.load_state_dict(dqn.state_dict())
    optimizer = optim.Adam(dqn.parameters(), lr=lr)    
    replay_buffer = dqn_agent.ReplayBuffer(buffer_capacity)
    
    # rb = PrioritizedReplayBuffer(alpha=0.7, beta=0.9, storage=ListStorage(buffer_capacity))
   

    def get_epsilon(step):
        est_total_steps = 52*num_episodes
        epsilon_decay = est_total_steps * epsilon_decay_prop
        return max(epsilon_end, epsilon_start - (step / epsilon_decay)*(epsilon_start - epsilon_end))

    total_steps = 0
    debug = False
    
    if not debug:
        wandb.init(
        # set the wandb project where this run will be logged
        project="yahtzee",
        # track hyperparameters and run metadata
        config={
        "learning_rate": lr,
        "episodes": num_episodes,
        "epsilon_decay": epsilon_decay_prop, 
        "buffer_capacity": buffer_capacity, 
        "batch_size": batch_size,
        })


    for episode in range(num_episodes):
        env = env_cls()
        #init environment with state, done status, and valid action mask
        state_dict = env.reset()
        state = env.get_encoded_state()
        state = torch.FloatTensor(state).to(device)
        done = False
        valid_actions_mask = env.get_valid_actions_mask()
        
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
                    
            #debug
            if debug:
                print("Dice before step: ", env.dice)
                print("Taking action:", action)
            
            # Execute action
            next_state, reward, done, _ = env.step(action)
            next_state = torch.FloatTensor(next_state).to(device)
            next_valid_mask = env.get_valid_actions_mask()
            
            #debug
            if debug:
                if action_idx >=32:
                    print("scored: ")
                    print(env.get_state()['categories'][category])
                
                print("Dice after action:", env.dice)
                print("\n")
            
            
            # Store experience in replay buffer
            replay_buffer.push(state.cpu(), action_idx, reward, next_state.cpu(), done, torch.tensor(next_valid_mask))
            
            state = next_state
            valid_actions_mask = next_valid_mask
            total_steps += 1

            # Training step
            if len(replay_buffer.buffer) >= batch_size:
                s_batch, a_batch, r_batch, s2_batch, d_batch, next_masks_batch = replay_buffer.sample(batch_size)
                
                # Convert to GPU tensors
                s_batch = s_batch.to(device)
                a_batch = a_batch.to(device)
                r_batch = r_batch.to(device)
                s2_batch = s2_batch.to(device)
                d_batch = d_batch.to(device)
                next_masks_batch = next_masks_batch.to(device)

                # Compute Q-values and loss
                q_values = dqn(s_batch)
                q_selected = q_values.gather(1, a_batch.unsqueeze(1)).squeeze(1)
                
                if (episode + 1) % eval_interval == 0:
                    if not debug:
                        wandb.log({"q_val_mean": torch.mean(q_values), "q_val_max":torch.max(q_values)}, step=episode+1)

                
                #debug
                # if total_steps % 1000 == 0:
                #     print(q_values[0])
                
                with torch.no_grad():
                    q_next = target_dqn(s2_batch)
                    # Mask invalid actions by setting their Q-values to -inf
                    q_next[~next_masks_batch.to(device).bool()] = -torch.inf
                    max_q_next = q_next.max(dim=1)[0]
                
                q_target = r_batch + gamma * max_q_next * (1 - d_batch)
                loss_function = nn.SmoothL1Loss()
                loss = loss_function(q_selected, q_target)
                
                if (episode + 1) % eval_interval == 0:
                    if not debug:
                        wandb.log({"reward_avg": torch.mean(r_batch)}, step=episode+1)


                optimizer.zero_grad()
                loss.backward()
                
                # Apply gradient clipping
                torch.nn.utils.clip_grad_norm_(dqn.parameters(), max_norm=max_grad_norm)  # Adjust max_norm as needed
                optimizer.step()

            # Update target network
            if total_steps % update_target_every == 0:
                target_dqn.load_state_dict(dqn.state_dict())
        
        if debug:
            final_score = sum(v for v in env.categories.values() if v is not None)
            print("final_score: ", final_score)
            print("end state: ", env.get_state())

        # Periodic evaluation
        if (episode + 1) % eval_interval == 0:
            avg_score = evaluate_model(dqn, env_cls, num_episodes=eval_episodes)
            print(f"Evaluation after episode {episode+1}: Average score over {eval_episodes} games = {avg_score:.1f}")
            if not debug:
                wandb.log({"avg_score": avg_score, "loss": loss, "epsilon": epsilon}, step=episode+1)

        # Episode statistics
        if (episode + 1) % 100 == 0:
            final_score = sum(v for v in env.categories.values() if v is not None)
            final_score += env.upper_bonus + env.yahtzee_bonuses
            print(f"Episode {episode+1}: Training score = {final_score:.1f}, Epsilon = {epsilon:.3f}")

    return dqn, target_dqn

if __name__ == "__main__":
    def make_env():
        return YahtzeeGame()

    dqn, target_dqn = train_dqn(make_env, 
                               num_episodes=10000,
                               eval_interval=100,
                               eval_episodes=100)
    
    # print(evaluate_model(dqn_agent.DQN(), make_env, num_episodes=1000, epsilon=1))
    
    

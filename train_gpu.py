

import utils
import dqn_agent 
import torch.optim as optim
import torch
import torch.nn as nn
from yahtzee import YahtzeeGame
import numpy as np
import random 
import wandb
from tensordict import TensorDict
from torchrl.data import ListStorage, PrioritizedReplayBuffer, LazyTensorStorage
import os
import itertools
from torch.optim.lr_scheduler import LambdaLR
import statistics
import yaml

ALL_ACTIONS = utils.generate_all_actions()
torch.manual_seed(1)
np.random.seed(1)
random.seed(1)

import torch
import statistics

def evaluate_model(model, env_cls, num_episodes=500, epsilon=0):
    """Evaluate the model's performance over multiple episodes without exploration."""
    model.eval()  # Set model to evaluation mode
    
    total_scores = 0.0
    scores = []

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
            scores.append(final_score)
    
    avg_score = total_scores / num_episodes
    median_score = statistics.median(scores)

    model.train()  # Set model back to training mode
    return avg_score, env.categories, median_score



def save_checkpoint(
    filename, 
    dqn, 
    target_dqn, 
    optimizer, 
    episode,
    total_steps
):
    checkpoint = {
        'dqn_state_dict': dqn.state_dict(),
        'target_dqn_state_dict': target_dqn.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'episode': episode,
        'total_steps': total_steps
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")


def load_checkpoint(filename, dqn, target_dqn, optimizer):
    if not os.path.isfile(filename):
        print(f"Checkpoint file '{filename}' does not exist. Starting fresh.")
        return 0, 0
    
    checkpoint = torch.load(filename, map_location=device)
    dqn.load_state_dict(checkpoint['dqn_state_dict'])
    target_dqn.load_state_dict(checkpoint['target_dqn_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    episode = checkpoint.get('episode', 0)
    total_steps = checkpoint.get('total_steps', 0)
    print(f"Loaded checkpoint from {filename} at episode={episode}, total_steps={total_steps}")
    return episode, total_steps


def train_dqn(env_cls, num_episodes=10000,
              buffer_capacity=5000,
              batch_size=256,
              gamma=0.99,
              lr=1e-4,
              epsilon_start=1.0,
              epsilon_end=0.1,
              epsilon_decay_prop=0.7,  # % of total steps
              update_target_every=1000,
              eval_interval=100,
              eval_episodes=100,
              max_grad_norm = 0.5,
              buffer_alpha = 0.7,
              buffer_beta = 0.9,
              save_checkpoint_dir=None,     
              load_checkpoint_path=None,
              debug = False,
              soft_tau = 0.01,  # or 0.001, 0.005, etc.
              hidden_dim = 256,
              ):
    """
    Train DQN agent with periodic evaluation.
    """
    # Initialize networks
    input_length = len(env_cls().get_encoded_state())
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

  # init dueling networks w target
    dqn = dqn_agent.DuelingDQN(state_dim=input_length, action_dim=len(ALL_ACTIONS), hidden_dim=hidden_dim).to(device)
    target_dqn = dqn_agent.DuelingDQN(state_dim=input_length, action_dim=len(ALL_ACTIONS), hidden_dim=hidden_dim).to(device)
    target_dqn.load_state_dict(dqn.state_dict())
    
    optimizer = optim.Adam(dqn.parameters(), lr=lr)    
    # lr decay scheduler: linear decay for 80% of training
    lr_lambda = lambda step: max( (lr/10) / lr, 1 - step / (num_episodes*0.8))
    scheduler = LambdaLR(optimizer, lr_lambda)  

    
   # Initialize the prioritized replay buffer
    storage = LazyTensorStorage(buffer_capacity)
    replay_buffer = PrioritizedReplayBuffer(
        alpha=buffer_alpha,  # Controls the degree of prioritization (0 is uniform sampling)
        beta=buffer_beta,   # Controls the amount of importance sampling correction
        storage=storage
    )
    


    def get_epsilon(step):
        est_total_steps = 52*num_episodes
        epsilon_decay = est_total_steps * epsilon_decay_prop
        return max(epsilon_end, epsilon_start - (step / epsilon_decay)*(epsilon_start - epsilon_end))

    # --- Optionally load from checkpoint ---
    start_episode = 0
    total_steps = 0
    if load_checkpoint_path is not None:
        start_episode, total_steps = load_checkpoint(
            load_checkpoint_path,
            dqn,
            target_dqn,
            optimizer
        )
    
    total_steps = 0
    
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
        "state_space_size": input_length,
        "update_target_every": update_target_every,
        "buffer_alpha": buffer_alpha,
        "buffer_beta": buffer_beta,
        "hidden_dim": hidden_dim, 
        })
        
        
        #set model name for saving
        save_checkpoint_path = save_checkpoint_dir + wandb.run.name
    
    max_med_score = 0

    for episode in range(num_episodes):
        env = env_cls()
        #init environment with state, done status, and valid action mask
        state_dict = env.reset()
        state = env.get_encoded_state()
        state = torch.FloatTensor(state).to(device)
        done = False
        valid_actions_mask = env.get_valid_actions_mask()
        
        while not done:
            
            #selection action, eps greedy
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
                    
            # Debugging statements
            if debug:
                print("Dice before step: ", env.dice)
            
            # Execute action
            next_state, reward, done, _ = env.step(action)
            # Debugging statements
            if debug:
                print("Action taken:", action)
                print("rerolls left: ", env.rolls_left)
            next_state = torch.FloatTensor(next_state).to(device)
            next_valid_mask = env.get_valid_actions_mask()
            
            # Debugging statements
            if debug and action_idx >=32:
                print("scored: ")
                print(env.get_state()['categories'][category])
                print("\n")
            
            # Store experience in replay buffer
            experience = TensorDict({
                'state': state.cpu(),
                'action': torch.tensor(action_idx),
                'reward': torch.tensor(reward),
                'next_state': next_state.cpu(),
                'done': torch.tensor(done),
                'next_valid_mask': torch.tensor(next_valid_mask)
            }, batch_size=[])
            
            replay_buffer.add(experience)
            
            state = next_state
            valid_actions_mask = next_valid_mask
            total_steps += 1

            # Sample a batch of experiences and perform training
            if len(replay_buffer) >= batch_size:
                batch, info = replay_buffer.sample(batch_size, return_info=True)

                # Extract experiences
                states = batch['state'].to(device)
                actions = batch['action'].to(device)
                rewards = batch['reward'].to(device)
                next_states = batch['next_state'].to(device)
                dones = batch['done'].to(device)
                next_valid_masks = batch['next_valid_mask'].to(device)

                # Extract sampling weights and indices
                weights = info['_weight'].clone().detach().to(dtype=torch.float32, device=device)
                indices = info['index']

                # Compute Q-values and gather the Q-values for taken actions
                q_values = dqn(states)  # Shape: [batch_size, action_dim]
                q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # Shape: [batch_size]

                if (episode + 1) % eval_interval == 0 and not debug:
                    wandb.log({
                        "q_val_mean": torch.mean(q_values).item(),
                        "q_val_max": torch.max(q_values).item()
                    }, step=episode+1)

                with torch.no_grad():
                    # Compute target Q-values using the target network
                    q_next = target_dqn(next_states)  # Shape: [batch_size, action_dim]
                    
                    # Mask invalid actions in next states
                    invalid_next_actions = ~next_valid_masks.to(device).bool()
                    q_next[invalid_next_actions] = -float('inf')

                    # Compute max Q-value for next states
                    max_q_next = q_next.max(dim=1)[0]  # Shape: [batch_size]

                    # Compute target: r + gamma * max(Q') * (1 - done)
                    q_target = rewards + gamma * max_q_next * (1 - dones.float())

                # Compute loss using Huber loss (Smooth L1)
                loss_function = nn.SmoothL1Loss(reduction='none')
                loss = loss_function(q_selected, q_target)

                # Apply importance sampling weights
                loss = (loss * weights).mean()

                if (episode + 1) % eval_interval == 0 and not debug:
                    wandb.log({
                        "reward_avg": torch.mean(rewards).item()
                    }, step=episode+1)

                # Compute TD-errors for priority update
                with torch.no_grad():
                    td_errors = torch.abs(q_target - q_selected).cpu()

                # Update priorities in the replay buffer
                replay_buffer.update_priority(indices, td_errors)

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()

                # Apply gradient clipping
                torch.nn.utils.clip_grad_norm_(dqn.parameters(), max_norm=max_grad_norm)
                optimizer.step()
                
                

                
                #for noisy networks
                # # Reset noise in both online and target networks
                # dqn.reset_noise()
                # target_dqn.reset_noise()

            # Update target network periodically -- hard update
            if total_steps % update_target_every == 0:
                target_dqn.load_state_dict(dqn.state_dict())
            
            # # soft update
            # for target_param, online_param in zip(target_dqn.parameters(), dqn.parameters()):
            #     target_param.data.copy_(
            #         soft_tau * online_param.data + (1.0 - soft_tau) * target_param.data)
    

        
        # update lr
        scheduler.step()
        
        if debug:
            final_score = sum(v for v in env.categories.values() if v is not None)
            print("final_score: ", final_score)
            print("end state: ", env.get_state())

        # Periodic evaluation
        if (episode + 1) % eval_interval == 0 and not debug:
            avg_score, categories, med_score = evaluate_model(dqn, env_cls, num_episodes=eval_episodes)
            print(f"Evaluation after episode {episode+1}: Score over {eval_episodes} games = Avg: {avg_score:.1f}, Med: {med_score:.1f}")
            wandb.log({"avg_score": avg_score, "med_score": med_score, "loss": loss.item(),"epsilon": epsilon}, step=episode+1)
            
            if med_score > max_med_score and episode > 10000:
                max_med_score = med_score
                if save_checkpoint_dir is not None:
                    save_checkpoint(
                        save_checkpoint_path + "_med_" + str(med_score), 
                        dqn,
                        target_dqn,
                        optimizer,
                        num_episodes,
                        total_steps
                    )
                

        # Episode statistics
        if (episode + 1) % 100 == 0:
            final_score = sum(v for v in env.categories.values() if v is not None)
            final_score += env.upper_bonus + env.yahtzee_bonuses
            print(f"Episode {episode+1}: Training score = {final_score:.1f}, Epsilon = {epsilon:.3f}, lr = {scheduler.get_last_lr()[0]:.7f}")

    # Final save at the end of training
    if save_checkpoint_dir is not None:
        save_checkpoint(
            save_checkpoint_path, 
            dqn,
            target_dqn,
            optimizer,
            num_episodes,
            total_steps
        )
        
    return dqn, target_dqn



if __name__ == "__main__":
    def make_env():
        return YahtzeeGame()
    
    # Check for GPU availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    pid = os.getpid()
    print(f"My process ID is: {pid}")
    print("device: ", device)
    
    params = {
    "num_episodes": 200000,
    "eval_interval": 100,
    "eval_episodes": 500,
    "save_checkpoint_dir": "/home/mc5635/yahtzee/yahtzee_rl/saved_models/",
    "load_checkpoint_path": None,  # "/home/mc5635/yahtzee/yahtzee_rl/saved_models/jumping-deluge-153_216.37"
    "lr": 0.0001,
    "epsilon_start": 1.0,
    "epsilon_decay_prop": 0.7,
    "buffer_beta": 0.7,
    "buffer_alpha": 0.7,
    "buffer_capacity": 10000,
    "batch_size": 256,
    "soft_tau": 0.001,
    "hidden_dim": 128,
    "update_target_every": 2000
}

    print(params)
    
    dqn, target_dqn = train_dqn(make_env, **params)
                               


# sweep call


    
def param_sweep():
    
    def make_env():
        return YahtzeeGame()
    
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    run = wandb.init()
    config = wandb.config
    
    pid = os.getpid()
    print(f"My process ID is: {pid}")
    print("device: ", device)
    
    config = wandb.config
    
    lr = config.lr
    epsilon_decay_prop = config.epsilon_decay_prop
    buffer_capacity = config.buffer_capacity
    batch_size = config.batch_size
    buffer_alpha = config.buffer_alpha
    buffer_beta = config.buffer_beta
    soft_tau = config.soft_tau
    max_grad_norm = config.max_grad_norm

    dqn, target_dqn = train_dqn(
                                make_env, 
                                num_episodes=12000,
                                eval_interval=100,
                                eval_episodes=500,
                                save_checkpoint_dir="/home/mc5635/yahtzee/yahtzee_rl/saved_models/",  
                                load_checkpoint_path=None, #"/home/mc5635/yahtzee/yahtzee_rl/saved_models/jumping-deluge-153_216.37",
                                lr=config.lr,
                                epsilon_decay_prop = 0.7,
                                buffer_capacity=config.buffer_capacity,
                                batch_size=config.batch_size,
                                buffer_alpha=config.buffer_alpha,
                                buffer_beta=config.buffer_beta, 
                                max_grad_norm=config.max_grad_norm,
                            
                            )
    

    
# if __name__ == "__main__":
#     # 1. Load your sweep configuration from a YAML file
#     with open("sweep_config.yaml", "r") as f:
#         sweep_config = yaml.safe_load(f)
    
#     # 2. Create the sweep on Weights & Biases
#     sweep_id = wandb.sweep(
#         sweep=sweep_config, 
#         project="my_yahtzee_dqn_project"  # Update with your project name
#     )
    
#     # 3. Start running the sweep agent
#     wandb.agent(
#         sweep_id=sweep_id, 
#         function=param_sweep, 
#         count=10  # Adjust for however many runs you want
#     )
    
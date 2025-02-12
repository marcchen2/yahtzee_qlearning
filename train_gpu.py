

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
    return avg_score, env.categories


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
              update_target_every=500,
              eval_interval=100,
              eval_episodes=100,
              max_grad_norm = 0.5,
              buffer_alpha = 0.7,
              buffer_beta = 0.9,
              save_checkpoint_dir=None,     
              load_checkpoint_path=None,
              debug = False
              ):
    """
    Train DQN agent with periodic evaluation.
    """
    # Initialize networks
    input_length = len(env_cls().get_encoded_state())

  # init dueling networks w target
    dqn = dqn_agent.DuelingDQN(state_dim=input_length, action_dim=len(ALL_ACTIONS)).to(device)
    target_dqn = dqn_agent.DuelingDQN(state_dim=input_length, action_dim=len(ALL_ACTIONS)).to(device)
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
        "buffer_beta": buffer_beta
        })
        
        #set model name for saving
        save_checkpoint_path = save_checkpoint_dir + wandb.run.name
    
    max_score_avg = 0

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

            # Update target network periodically
            if total_steps % update_target_every == 0:
                target_dqn.load_state_dict(dqn.state_dict())
        
        # update lr
        scheduler.step()
        
        if debug:
            final_score = sum(v for v in env.categories.values() if v is not None)
            print("final_score: ", final_score)
            print("end state: ", env.get_state())

        # Periodic evaluation
        if (episode + 1) % eval_interval == 0 and not debug:
            avg_score, categories = evaluate_model(dqn, env_cls, num_episodes=eval_episodes)
            print(f"Evaluation after episode {episode+1}: Average score over {eval_episodes} games = {avg_score:.1f}")
            wandb.log({"avg_score": avg_score, "loss": loss.item(),"epsilon": epsilon}, step=episode+1)
            
            if avg_score > max_score_avg:
                max_score_avg = avg_score
                if save_checkpoint_dir is not None:
                    save_checkpoint(
                        save_checkpoint_path + "_" + str(avg_score), 
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
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


    dqn, target_dqn = train_dqn(make_env, 
                               num_episodes=100000,
                               eval_interval=100,
                               eval_episodes=100,
                               save_checkpoint_dir="/home/mc5635/yahtzee/yahtzee_rl/saved_models/",  
                               load_checkpoint_path= "/home/mc5635/yahtzee/yahtzee_rl/saved_models/jumping-deluge-153_216.37",
                               lr= 0.00001,
                               epsilon_start=0.15,
                               buffer_beta=0.3, 
                               buffer_capacity=10000,
                               batch_size=512,
                               )
    
    
    
    
    
    # check random agent score
    # print(evaluate_model(dqn_agent.DQN(), make_env, num_episodes=1000, epsilon=1))
    
    

    
    
    
def hyperparam_sweep():
    # Define possible values for hyperparameters

    buffer_alpha = [0.7,0.5]
    buffer_beta = [0.9,0.5]
    buffer_capacity = [5000, 10000]

    # You can add more or fewer hyperparams to sweep over, e.g.:
    # epsilon_decay_props = [0.5, 0.8]

    # Create a simple combinatorial grid
    for (buffer_alpha, buffer_beta, buffer_capacity) in itertools.product(buffer_alpha, buffer_beta, buffer_capacity

    ):
        # Optionally create a config dict
        config = {
            "buffer_alpha": buffer_alpha,
            "buffer_beta": buffer_beta,
            "buffer_capacity": buffer_capacity,
            
        }
        
        # --- Start a new W&B run for each hyperparam combo ---
        wandb.init(
            project="yahtzee-hyperparam-sweep3",
            config=config,
            reinit=True  # Ensures each loop iteration is a fresh run
        )

        # Optionally, you can pass these directly to train_dqn as kwargs
        def make_env():
            return YahtzeeGame()  # Or whatever your environment constructor is

        dqn, target_dqn = train_dqn(
            env_cls=make_env,
            num_episodes=5000,      # You can reduce for quick testing
            eval_interval=100,
            eval_episodes=100,
            save_checkpoint_dir="/home/mc5635/yahtzee/yahtzee_rl/param_sweep_models/",  # or your directory
            load_checkpoint_path=None,
            buffer_alpha=buffer_alpha, 
            buffer_beta=buffer_beta,
            buffer_capacity=buffer_capacity,
            
        )

        # Mark the run finished so a new one starts in the next iteration
        wandb.finish()

# if __name__ == "__main__":
#     # Check for GPU availability
#     device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

#     hyperparam_sweep()


import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import wandb

# Local imports (adjust as needed)
import toy_dqn as dqn_agent
from toy_yahtzee import SimpleYahtzeeGame

def generate_all_actions():
    """
    Return a list of action indices [0..12].
    (In your original code, you had 45 possible actions, 
    but here we only have 13 for the simpler environment.)
    """
    return list(range(13))



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ALL_ACTIONS = generate_all_actions()  # [0..12]

def evaluate_model(model, env_cls, num_episodes=100, epsilon=0):
    """Evaluate the model without exploration."""
    model.eval()
    total_scores = 0.0
    
    with torch.no_grad():
        for _ in range(num_episodes):
            env = env_cls()
            env.reset()
            done = False
            
            while not done:
                state = torch.FloatTensor(env.get_encoded_state()).to(device)
                valid_mask = env.get_valid_actions_mask()
                action_idx = dqn_agent.select_action(
                    model, state, valid_mask, epsilon
                )

                if action_idx < 8:
                    # Convert index => 3-bit mask
                    keep_mask = [bool(int(bit)) for bit in f"{action_idx:03b}"]
                    action = ('reroll', keep_mask)
                else:
                    cat_idx = action_idx - 8  # shift by 8
                    category = list(env.categories.keys())[cat_idx]
                    action = ('score', category)
                
                _, _, done, _ = env.step(action)

            # Calculate final score
            # No upper_bonus or yahtzee_bonuses in SimpleYahtzeeGame
            final_score = sum(v for v in env.categories.values() if v is not None)
            total_scores += final_score
    
    model.train()
    return total_scores / num_episodes

def train_dqn(env_cls, 
              num_episodes=10000,
              buffer_capacity=1000,
              batch_size=64,
              gamma=0.99,
              lr=1e-4,
              epsilon_start=1.0,
              epsilon_end=0.1,
              epsilon_decay=50000,
              update_target_every=100,
              eval_interval=1000,
              eval_episodes=100):
    
    # Initialize DQN
    dqn = dqn_agent.DQN(state_dim=9, action_dim=13, hidden=64).to(device)
    target_dqn = dqn_agent.DQN(state_dim=9, action_dim=13, hidden=64).to(device)
    target_dqn.load_state_dict(dqn.state_dict())
    optimizer = optim.Adam(dqn.parameters(), lr=lr)
    replay_buffer = dqn_agent.ReplayBuffer(buffer_capacity)

    def get_epsilon(step):
        return max(epsilon_end,
                   epsilon_start - (step / epsilon_decay)*(epsilon_start - epsilon_end))

    wandb.init(project="toy-yahtzee",
               config={
                   "learning_rate": lr,
                   "episodes": num_episodes,
                   "epsilon_decay": epsilon_decay,
                   "batch_size": batch_size,
                   "buffer_capacity": buffer_capacity,
               })

    total_steps = 0
    for episode in range(num_episodes):
        env = env_cls()
        env.reset()
        
        state = torch.FloatTensor(env.get_encoded_state()).to(device)
        valid_actions_mask = env.get_valid_actions_mask()
        done = False

        while not done:
            epsilon = get_epsilon(total_steps)
            action_idx = dqn_agent.select_action(dqn, state, valid_actions_mask, epsilon)

            # Convert action_idx -> actual action
            if action_idx < 8:
                keep_mask = [bool(int(bit)) for bit in f"{action_idx:03b}"]
                action = ('reroll', keep_mask)
            else:
                cat_idx = action_idx - 8
                category = list(env.categories.keys())[cat_idx]
                action = ('score', category)

            next_state, reward, done, _ = env.step(action)
            next_state_t = torch.FloatTensor(next_state).to(device)
            next_valid_mask = env.get_valid_actions_mask()

            # Store transition in buffer (CPU tensors)
            replay_buffer.push(
                state.cpu(), 
                action_idx, 
                reward, 
                next_state_t.cpu(),
                done
            )

            state = next_state_t
            valid_actions_mask = next_valid_mask
            total_steps += 1

            # If enough transitions, start training
            if len(replay_buffer.buffer) >= batch_size:
                s_batch, a_batch, r_batch, s2_batch, d_batch = replay_buffer.sample(batch_size)

                q_values = dqn(s_batch)                # [B, 13]
                q_chosen = q_values.gather(1, a_batch.unsqueeze(1)).squeeze(1)  # [B]

                with torch.no_grad():
                    q_next = target_dqn(s2_batch)      # [B, 13]
                    max_q_next = q_next.max(dim=1)[0]  # [B]

                target = r_batch + gamma * max_q_next * (1 - d_batch)
                loss = nn.MSELoss()(q_chosen, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Periodically update target net
            if episode % update_target_every == 0:
                target_dqn.load_state_dict(dqn.state_dict())

        # Periodic evaluation
        if (episode + 1) % eval_interval == 0:
            avg_score = evaluate_model(dqn, env_cls, num_episodes=eval_episodes, epsilon=0.0)
            print(f"Episode {episode+1}, avg_score over {eval_episodes} games = {avg_score:.2f}")
            wandb.log({"episode": episode+1, "avg_score": avg_score, "loss": loss}, step=episode+1)

    return dqn, target_dqn


if __name__ == "__main__":
    def make_env():
        return SimpleYahtzeeGame()

    dqn, target_dqn = train_dqn(
        env_cls=make_env,
        num_episodes=20000,
        eval_interval=1000,
        eval_episodes=100,
        batch_size=64,
        buffer_capacity=20000
    )
    
    # dqn = dqn_agent.DQN()
    # Optionally evaluate final model:
    final_eval = evaluate_model(dqn, make_env, num_episodes=500)
    print(f"Final Evaluation Score (500 games): {final_eval:.2f}")

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
    """
    env_cls: a callable that returns a new YahtzeeGame() instance
    num_episodes: how many full games to play
    buffer_capacity: replay buffer size
    batch_size: mini-batch size
    gamma: discount factor
    lr: learning rate
    epsilon_*: params for epsilon schedule
    update_target_every: how often to sync target net
    """

    # Create DQN, target net, replay buffer
    dqn = dqn_agent.DQN(state_dim=40, action_dim=len(ALL_ACTIONS))
    target_dqn = dqn_agent.DQN(state_dim=40, action_dim=len(ALL_ACTIONS))
    target_dqn.load_state_dict(dqn.state_dict())
    optimizer = optim.Adam(dqn.parameters(), lr=lr)
    replay_buffer = dqn_agent.ReplayBuffer(buffer_capacity)

    # Simple linear epsilon schedule
    def get_epsilon(step):
        return max(epsilon_end, epsilon_start - (step / epsilon_decay)*(epsilon_start - epsilon_end))

    total_steps = 0  # count total environment steps

    for episode in range(num_episodes):
        # New game
        env = env_cls()
        state_dict = env.reset()
        # We'll get the encoded state as a torch tensor
        state = env.get_encoded_state()  # shape (40,)

        done = False
        # A single game can have up to 13 category-scoring actions, but let's just loop until done
        while not done:
            # Epsilon-greedy action
            eps = get_epsilon(total_steps)
            action_idx = dqn_agent.select_action(dqn, state, eps)
            # Convert action_idx -> actual environment action
            action = ALL_ACTIONS[action_idx]

            # Step environment
            next_state_dict, reward, done, _info = env.step(action)
            next_state = env.get_encoded_state()

            # Store transition
            replay_buffer.push(state, action_idx, reward, next_state, done)
            state = next_state
            total_steps += 1
            
            # #print debugging
            # print(f"Chose action_idx = {action_idx}, action = {ALL_ACTIONS[action_idx]}, reward = {reward}")
            # print(f"Next state dice = {next_state_dict['dice']}, categories = {next_state_dict['categories']}")


            # DQN update step
            if len(replay_buffer) >= batch_size:
                # 1) Sample batch
                s_batch, a_batch, r_batch, s2_batch, d_batch = replay_buffer.sample(batch_size)

                # 2) Compute Q(s,a)
                q_values = dqn(s_batch)       # shape: (batch, 45)
                q_selected = q_values.gather(1, a_batch.view(-1,1)).squeeze(1)  # shape: (batch,)

                # 3) Compute target = r + gamma * max Q(s2, ·), but 0 if done
                with torch.no_grad():
                    # Compute max action for s2 using target net
                    q_next = target_dqn(s2_batch)  # shape: (batch, 45)
                    max_q_next = q_next.max(dim=1)[0]  # shape: (batch,)

                q_target = r_batch + gamma * max_q_next * (1 - d_batch)

                # 4) Loss
                loss = nn.MSELoss()(q_selected, q_target)

                # 5) Backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Occasionally update target network
            if total_steps % update_target_every == 0:
                target_dqn.load_state_dict(dqn.state_dict())

        # End of the episode
        # Optionally, log scores or track stats here
        # e.g. final score from env
        final_score = sum(v for v in env.categories.values() if v is not None)
        final_score += env.upper_bonus + env.yahtzee_bonuses
        if (episode+1) % 100 == 0:
            print(f"Episode {episode+1}: final_score = {final_score:.1f}, epsilon = {eps:.3f}")

    return dqn, target_dqn

def evaluate_agent(dqn, env_cls, ALL_ACTIONS, n_eval_episodes=100, max_steps_per_episode=50):
    """
    Evaluate the trained DQN agent over n_eval_episodes with epsilon=0 (greedy policy).
    Returns average and median final scores including bonuses.
    
    Args:
        dqn: Trained DQN model
        env_cls: Callable that returns a YahtzeeGame instance
        ALL_ACTIONS: List of all possible actions
        n_eval_episodes: Number of evaluation episodes
        max_steps_per_episode: Safety limit for steps per episode
    
    Returns:
        (avg_score, median_score): Tuple of average and median scores
    """
    dqn.eval()  # Set network to evaluation mode
    scores = []
    
    for episode in range(n_eval_episodes):
        env = env_cls()
        env.reset()
        state = env.get_encoded_state()  # Initial encoded state
        done = False
        steps = 0
        
        while not done and steps < max_steps_per_episode:
            with torch.no_grad():
                q_values = dqn(state.unsqueeze(0))  # Add batch dimension
                action_idx = q_values.argmax().item()
            
            action = ALL_ACTIONS[action_idx]
            _, reward, done, _ = env.step(action)
            state = env.get_encoded_state()  # Update to new state
            steps += 1
        
        # Calculate final score with bonuses
        final_score = sum(v for v in env.categories.values() if v is not None)
        final_score += env.upper_bonus + env.yahtzee_bonuses
        scores.append(final_score)
        
        # Progress reporting
        if (episode+1) % 10 == 0:
            print(f"Episode {episode+1}/{n_eval_episodes} | Score: {final_score}")

    avg_score = np.mean(scores)
    median_score = np.median(scores)
    print(f"\nEvaluation Results ({n_eval_episodes} episodes):")
    print(f"Average Score: {avg_score:.1f}")
    print(f"Median Score: {median_score:.1f}")
    print(f"Minimum Score: {np.min(scores):.1f}")
    print(f"Maximum Score: {np.max(scores):.1f}")
    
    return avg_score, median_score



if __name__ == "__main__":
    # We'll pass a lambda that returns a fresh YahtzeeGame.
    def make_env():
        return YahtzeeGame()

    dqn, target_dqn = train_dqn(make_env, num_episodes=10000)
    
    # Then test or evaluate the trained dqn
    # evaluate_agent(dqn, YahtzeeGame, ALL_ACTIONS, n_eval_episodes=100)
    
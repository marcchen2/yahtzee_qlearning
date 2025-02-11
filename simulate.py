import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import gym  # Assuming your Yahtzee environment follows OpenAI Gym API
import seaborn as sns
import yahtzee
import dqn_agent
# Load your environment
env = yahtzee.YahtzeeGame()  # Adjust this to your actual environment

# Load the trained model
class AgentModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AgentModel, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 128)
        self.fc2 = dqn_agent.NoisyLinear(128, 128)  # Using your NoisyLinear layer
        self.fc3 = torch.nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Define input and output sizes based on your environment
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n

# Instantiate and load model
model = dqn_agent.DuelingDQN()
model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
model.eval()

# Function to visualize episode
def visualize_episode(env, model, num_episodes=1):
    rewards_per_step = []
    actions_taken = []
    state_values = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_rewards = 0
        step = 0

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_probs = model(state_tensor)
            action = torch.argmax(action_probs, dim=1).item()

            next_state, reward, done, _ = env.step(action)

            # Store data for visualization
            rewards_per_step.append(reward)
            actions_taken.append(action)
            state_values.append(state)

            state = next_state
            episode_rewards += reward
            step += 1

    # Plot rewards per step
    plt.figure(figsize=(12, 6))
    plt.plot(rewards_per_step, label="Reward per Step")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title("Agent Rewards Over Time")
    plt.legend()
    plt.show()

    # Plot action distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(actions_taken, bins=env.action_space.n, kde=False)
    plt.xlabel("Action")
    plt.ylabel("Frequency")
    plt.title("Distribution of Actions Taken")
    plt.show()

    # Visualizing state space evolution (if low-dimensional)
    if len(state_values[0]) <= 2:  # Only plot if state space is small (2D)
        states_np = np.array(state_values)
        plt.figure(figsize=(8, 8))
        plt.scatter(states_np[:, 0], states_np[:, 1], c=np.arange(len(states_np)), cmap="viridis")
        plt.xlabel("State Dimension 1")
        plt.ylabel("State Dimension 2")
        plt.title("State Transitions Over Time")
        plt.colorbar(label="Time Step")
        plt.show()

# Run visualization
visualize_episode(env, model)

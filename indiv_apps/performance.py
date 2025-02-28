import gradio as gr
import torch
import numpy as np
from yahtzee import YahtzeeGame
import utils
import dqn_agent


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALL_ACTIONS = utils.generate_all_actions()

# Load trained model
def load_trained_model(checkpoint_path, model_class, device='cpu'):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    dqn_state_dict = checkpoint['dqn_state_dict']

    state_dim, hidden_dim, action_dim = None, None, None

    for key, tensor in dqn_state_dict.items():
        if "feature_layer.0.weight" in key:
            state_dim = tensor.shape[1]
            hidden_dim = tensor.shape[0]
        elif "advantage_stream.weight" in key:
            action_dim = tensor.shape[0]

    if state_dim is None or hidden_dim is None or action_dim is None:
        raise ValueError("Failed to extract model dimensions from checkpoint.")

    model = model_class(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(dqn_state_dict)
    model.eval()
    
    return model

trained_model = load_trained_model("/home/mc5635/yahtzee/yahtzee_rl/saved_models/riveting-valentine-200_med_211.0", dqn_agent.DuelingDQN)


def simulate_games(num_games):
    scores = []
    
    for _ in range(num_games):
        env = YahtzeeGame()
        state = torch.FloatTensor(env.get_encoded_state()).to(device)
        done = False

        while not done:
            valid_actions_mask = env.get_valid_actions_mask()
            action_idx = dqn_agent.select_action(trained_model, state, valid_actions_mask, epsilon=0.0)
            
            if action_idx < 32:
                keep_mask = [bool(int(bit)) for bit in f"{action_idx:05b}"]
                action = ('reroll', keep_mask)
            else:
                category = list(env.categories.keys())[action_idx - 32]
                action = ('score', category)
            
            next_state, _, done, _ = env.step(action)
            state = torch.FloatTensor(next_state).to(device)

        final_score = sum(v for v in env.categories.values() if v is not None) + env.upper_bonus + env.yahtzee_bonuses
        scores.append(final_score)
    
    mean_score = np.mean(scores)
    median_score = np.median(scores)
    
    return f"Mean Score: {mean_score:.2f}\nMedian Score: {median_score:.2f}"

# Gradio UI
iface = gr.Interface(
    fn=simulate_games,
    inputs=gr.Number(label="Number of Games", value=100, precision=0),
    outputs=gr.Textbox(label="Performance Metrics"),
    title="Marc's Yahtzee Agent Performance Calculator",
    description="Enter the number of games to simulate and evaluate the trained Yahtzee agent's performance."
)

iface.launch()

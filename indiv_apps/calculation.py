import gradio as gr
import torch
import pandas as pd
from yahtzee import YahtzeeGame
import utils
import dqn_agent
import numpy as np

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

trained_model = load_trained_model(
    "/home/mc5635/yahtzee/yahtzee_rl/saved_models/wise-armadillo-155_219.54", 
    dqn_agent.DuelingDQN, 
    device=device  # Add this to use the global device variable
)

def format_keep_mask(mask):
    """Formats the boolean reroll mask as [KK__K] where K=True, _=False."""
    return "[" + "".join("K" if x else "_" for x in mask) + "]"

def get_q_values(dice, categories, rolls_left):
    env = YahtzeeGame()
    env.dice = dice
    env.categories.update(categories)
    env.rolls_left = rolls_left
    state = torch.FloatTensor(env.get_encoded_state()).to(device)
    valid_actions_mask = np.array(env.get_valid_actions_mask(), dtype=bool)
    
    if len(state.shape) == 1:
        state = state.unsqueeze(0)  # Shape: [1, state_dim]
    
    with torch.no_grad():
        q_values = trained_model(state).cpu().numpy()
        q_values = q_values.flatten()
    
    # Apply action masking
    if rolls_left == 0:
        # Disable all reroll actions
        valid_actions_mask[:32] = False
        
        # Manually enable scoring actions for available categories
        for i, action in enumerate(ALL_ACTIONS):
            if action[0] == 'score':
                category_name = action[1]
                # Check if the category is available (value is None)
                if categories.get(category_name) is None:
                    valid_actions_mask[i] = True
                else:
                    valid_actions_mask[i] = False
    
    q_values[~valid_actions_mask] = -float('inf')
    
    action_values = {
        (action[0], format_keep_mask(action[1]) if isinstance(action[1], list) else action[1]): q_values[i]
        for i, action in enumerate(ALL_ACTIONS)
        if valid_actions_mask[i]
    }
    
    return action_values

def display_q_values(dice, categories, rolls_left):
    q_values = get_q_values(dice, categories, rolls_left)
    df = pd.DataFrame(q_values.items(), columns=["Action", "Expected Value"])
    df = df.sort_values(by="Expected Value", ascending=False)
    return df

iface = gr.Blocks()

with iface:
    gr.Markdown("<h1 style='color: orange;'>Marc's Yahtzee Q-Value Calculator</h1>")
    dice_input = gr.Textbox(label="Dice (comma-separated, e.g., 1,2,3,4,5)")
    rolls_left_input = gr.Number(label="Rolls Left", value=2)
    
    category_labels = ["Ones", "Twos", "Threes", "Fours", "Fives", "Sixes", "Three of a Kind", "Four of a Kind", "Full House", "Small Straight", "Large Straight", "Yahtzee", "Chance"]
    category_steps = {"Ones": 1, "Twos": 2, "Threes": 3, "Fours": 4, "Fives": 5, "Sixes": 6, "Three of a Kind": 1, "Four of a Kind": 1, "Full House": 25, "Small Straight": 30, "Large Straight": 40, "Yahtzee": 50, "Chance": 1}
    
    category_inputs = {}
    category_checkboxes = {}
    
    with gr.Row():
        for label in category_labels[:7]:
            with gr.Column():
                category_checkboxes[label] = gr.Checkbox(label=f"{label}", value=False)
                category_inputs[label] = gr.Number(label=label, value=None, step=category_steps[label])
    with gr.Row():
        for label in category_labels[7:]:
            with gr.Column():
                category_checkboxes[label] = gr.Checkbox(label=f"{label}", value=False)
                category_inputs[label] = gr.Number(label=label, value=None, step=category_steps[label])
    
    calculate_button = gr.Button("Calculate Expected Values")
    q_value_table = gr.Dataframe(label="Action Expected Values")
    
    def process_inputs(dice_str, rolls_left, *category_values):
        dice = list(map(int, dice_str.split(',')))
        categories = {label: (value if checkbox else None) for label, value, checkbox in zip(category_labels, category_values[:len(category_labels)], category_values[len(category_labels):])}
        return display_q_values(dice, categories, rolls_left)
    
    inputs = [dice_input, rolls_left_input] + list(category_inputs.values()) + list(category_checkboxes.values())
    calculate_button.click(fn=process_inputs, inputs=inputs, outputs=q_value_table)

iface.launch()

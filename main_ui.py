import gradio as gr
import numpy as np
import torch
import time
from yahtzee import YahtzeeGame
import utils
import dqn_agent
import pandas as pd
import os
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALL_ACTIONS = utils.generate_all_actions()

# Load trained model
def load_trained_model(checkpoint_path, model_class, device=device):
    """
    Loads a trained DuelingDQN model from a checkpoint file.

    Args:
        checkpoint_path (str): Path to the saved checkpoint file.
        model_class: The DuelingDQN model class.
        device (str): Device to load the model onto ('cpu' or 'cuda').

    Returns:
        model: The loaded DuelingDQN model.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    dqn_state_dict = checkpoint['dqn_state_dict']

    # Extract dimensions dynamically
    state_dim, hidden_dim, action_dim = None, None, None

    for key, tensor in dqn_state_dict.items():
        if "feature_layer.0.weight" in key:  # First Linear layer in feature extraction
            state_dim = tensor.shape[1]  # Input dimension
            hidden_dim = tensor.shape[0]  # First hidden layer size
        elif "advantage_stream.weight" in key:  # Last Linear layer in advantage stream
            action_dim = tensor.shape[0]  # Output dimension (number of actions)

    if state_dim is None or hidden_dim is None or action_dim is None:
        raise ValueError("Failed to extract model dimensions from checkpoint. Please check the saved model.")

    print(f"Detected model architecture: state_dim={state_dim}, action_dim={action_dim}, hidden_dim={hidden_dim}")

    # Initialize model with correct dimensions
    model = model_class(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(dqn_state_dict)
    model.eval()  # Set model to evaluation mode

    print(f"Loaded trained model from {checkpoint_path}")
    return model

# load model
script_dir = Path(__file__).parent
checkpoint_path = script_dir / "saved_models" / "wise-armadillo-155_228.27"
trained_model = load_trained_model(checkpoint_path, dqn_agent.DuelingDQN)

### simulation mode functions ###

# Initialize game
def initialize_game():
    global env, state, done, turn, rolls_this_turn, log_history
    env = YahtzeeGame()
    state_dict = env.reset()
    state = torch.FloatTensor(env.get_encoded_state()).to(device)
    done = False
    turn = 0
    rolls_this_turn = 0
    log_history = "Game Reset. Click 'Next Move' to start.\n"

initialize_game()

def format_log(log):
    return f"""
    <div id="game-log" style="height: 700px; overflow-y: auto; border: 1px solid #ccc; padding: 10px; font-family: monospace; white-space: pre-wrap;">
        {log}
    </div>
    """

def format_turn_indicator(turn_text):
    return f"<span style='color: orange; font-weight: bold;'>{turn_text}</span>"

def format_score_sheet(categories, upper_bonus=0, yahtzee_bonuses=0):
    df = pd.DataFrame(list(categories.items()), columns=["Category", "Score"])
    df.loc[len(df)] = ["Upper Bonus", upper_bonus]
    df.loc[len(df)] = ["Yahtzee Bonuses", yahtzee_bonuses]
    return df
       

def step_game():
    global state, done, turn, rolls_this_turn, log_history

    prev_upper_bonus = env.upper_bonus
    prev_yahtzee_bonuses = env.yahtzee_bonuses

    if done:
        return format_log(log_history + "\nGame Over. Reset to play again."), format_score_sheet(env.categories), pd.DataFrame()

    if rolls_this_turn == 0:
        turn += 1
        display = format_turn_indicator(f"\n===== TURN {turn} =====\n")
    else:
        display = ""

    rolls_this_turn += 1
    display += f"Roll {rolls_this_turn}: Dice = {env.dice}\n"

    # Get valid actions
    valid_actions_mask = env.get_valid_actions_mask()
    action_idx = dqn_agent.select_action(trained_model, state, valid_actions_mask, epsilon=0.0)

    # Calculate Q-values for display
    q_values = get_q_values(env.dice, env.categories, env.rolls_left)
    q_values_df = pd.DataFrame(q_values.items(), columns=["Action", "Expected Value"])
    q_values_df = q_values_df.sort_values(by="Expected Value", ascending=False)

    if action_idx < 32:  # Reroll action
        keep_mask = [bool(int(bit)) for bit in f"{action_idx:05b}"]
        formatted_keep_mask = "[" + "".join("K" if keep else "_" for keep in keep_mask) + "]"
        action = ('reroll', keep_mask)
        display += f"Rerolling, keeping: {formatted_keep_mask}\n"

    else:  # Scoring action
        category = list(env.categories.keys())[action_idx - 32]
        action = ('score', category)
        display += f"Scoring in category: {category}\n"

    prev_categories = env.categories.copy()

    next_state, reward, done, _ = env.step(action)
    state = torch.FloatTensor(next_state).to(device)

    if action[0] == "score":
        scored_value = env.categories[category] if env.categories[category] is not None else 0
        prev_value = prev_categories[category] if prev_categories[category] is not None else 0
        score_earned = scored_value - prev_value
        display += f"→ Scored {score_earned} points in {category}\n"
        rolls_this_turn = 0

    if env.upper_bonus > prev_upper_bonus:
        display += f"→ Upper Bonus Scored: {env.upper_bonus} points!\n"
    if env.yahtzee_bonuses > prev_yahtzee_bonuses:
        display += f"→ Yahtzee Bonus Scored: {env.yahtzee_bonuses} points!\n"

    if done:
        final_score = sum(v for v in env.categories.values() if v is not None) + env.upper_bonus + env.yahtzee_bonuses
        display += format_turn_indicator(f"\n===== FINAL SCORE: {final_score} =====\n")

    log_history += display
    return format_log(log_history), format_score_sheet(env.categories, env.upper_bonus, env.yahtzee_bonuses), q_values_df

def reset_game():
    initialize_game()
    return format_log(log_history), format_score_sheet(env.categories, env.upper_bonus, env.yahtzee_bonuses), pd.DataFrame()

####


### performance mode functions ###

def simulate_games(num_games):
    env = YahtzeeGame()  # Initialize the environment
    scores = []
    category_scores = {category: [] for category in env.categories.keys()}
    upper_bonus_scores = []
    yahtzee_bonus_scores = []
    
    for _ in range(num_games):
        env = YahtzeeGame()  # Re-initialize the environment for each game
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
        
        # Collect category scores and bonuses
        for category, score in env.categories.items():
            if score is not None:
                category_scores[category].append(score)
        upper_bonus_scores.append(env.upper_bonus)
        yahtzee_bonus_scores.append(env.yahtzee_bonuses)
    
    mean_score = np.mean(scores)
    median_score = np.median(scores)
    
    # Calculate average scores for each category and bonuses
    avg_category_scores = {category: np.mean(scores) if scores else 0 for category, scores in category_scores.items()}
    avg_upper_bonus = np.mean(upper_bonus_scores)
    avg_yahtzee_bonus = np.mean(yahtzee_bonus_scores)
    
    # Format the output
    category_scores_str = "\n".join([f"{category}: {avg_score:.2f}" for category, avg_score in avg_category_scores.items()])
    return (
        f"Mean Score: {mean_score:.2f}\n"
        f"Median Score: {median_score:.2f}\n"
        f"Average Category Scores:\n{category_scores_str}\n"
        f"Average Upper Bonus: {avg_upper_bonus:.2f}\n"
        f"Average Yahtzee Bonus: {avg_yahtzee_bonus:.2f}"
    )

####


### calculation mode functions ###


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

####

# Integrating the provided Gradio UI as one of the apps
def yahtzee_q_learning_app():
    iface = gr.Blocks()
    with iface:
        initialize_game()  # Ensure the game is reset upon loading
        gr.Markdown("<h1 style='color: orange;'>Game Simulator</h1>")

        with gr.Row():
            game_log = gr.HTML(label="Game Log", value=format_log(log_history), elem_id="game-log-container")
            score_sheet = gr.Dataframe(
                label="Scoring Sheet",
                value=format_score_sheet(env.categories, env.upper_bonus, env.yahtzee_bonuses),
                max_height=700
            )
            q_values_table = gr.Dataframe(label="Q-Values", value=pd.DataFrame(), max_height=700)

        next_move_button = gr.Button("Next Move")
        reset_button = gr.Button("Reset Game")
    
        # Attach JavaScript to scroll down after each update
        js_scroll = """
        function scrollLog() {
            let log = document.getElementById('game-log-container');
            if (log) {
                log.scrollTop = log.scrollHeight;
            }
        }
        setTimeout(scrollLog, 100);
        """

        next_move_button.click(fn=step_game, inputs=[], outputs=[game_log, score_sheet, q_values_table]).then(
            lambda: None, None, None, js=js_scroll
        )

        reset_button.click(fn=reset_game, inputs=[], outputs=[game_log, score_sheet, q_values_table]).then(
            lambda: None, None, None, js=js_scroll
        )
    
    return iface

def performance_mode():
    # Gradio UI
    iface = gr.Blocks()

    with iface:
        gr.Markdown("<h1 style='color: orange;'>Agent Performance Calculator</h1>")
        gr.Interface(
            fn=simulate_games,
            inputs=gr.Number(label="Number of Games", value=100, precision=0),
            outputs=gr.Textbox(label="Performance Metrics"),
            description="Enter the number of games to simulate and evaluate the trained Yahtzee agent's performance."
        )

    return iface

def calc_mode():
    iface = gr.Blocks()

    with iface:
        gr.Markdown("<h1 style='color: orange;'>Q-Value Calculator</h1></br>")
        gr.Markdown("<h2 style='color: white;'>Set State Space:</h2>")
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

    return iface

with gr.Blocks() as demo:
    gr.Markdown("<h1 style='color: orange;'>Marc's Q-Learning Agent</h1>")
    with gr.Tabs():
        with gr.Tab("Game Simulator"):
            yahtzee_q_learning_app()
        with gr.Tab("Performance Mode"):
            performance_mode()
        with gr.Tab("Calculation Mode"):
            calc_mode()

demo.launch(share=False)

import gradio as gr
import torch
import time
from yahtzee import YahtzeeGame
import utils
import dqn_agent
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALL_ACTIONS = utils.generate_all_actions()

# Load trained model
def load_trained_model(checkpoint_path, model_class, device='cpu'):
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

trained_model = load_trained_model("/home/mc5635/yahtzee/yahtzee_rl/saved_models/jumping-deluge-153_218.18", dqn_agent.DuelingDQN)

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

    prev_upper_bonus = env.upper_bonus  # Ensure these are assigned before usage
    prev_yahtzee_bonuses = env.yahtzee_bonuses

    if done:
        return format_log(log_history + "\nGame Over. Reset to play again."), format_score_sheet(env.categories)
    
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
    return format_log(log_history), format_score_sheet(env.categories, env.upper_bonus, env.yahtzee_bonuses)

def reset_game():
    initialize_game()
    return format_log(log_history), format_score_sheet(env.categories, env.upper_bonus, env.yahtzee_bonuses)

# Gradio UI

iface = gr.Blocks()

with iface:
    initialize_game()  # Ensure the game is reset upon loading
    gr.Markdown("<h1 style='color: orange;'>Marc's Yahtzee Q Learning Simulator</h1>")

    with gr.Row():
        game_log = gr.HTML(label="Game Log", value=format_log(log_history), elem_id="game-log-container")
        score_sheet = gr.Dataframe(
            label="Scoring Sheet",
            value=format_score_sheet(env.categories, env.upper_bonus, env.yahtzee_bonuses),
            max_height=700
        )

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

    next_move_button.click(fn=step_game, inputs=[], outputs=[game_log, score_sheet]).then(
        lambda: None, None, None, js=js_scroll
    )

    reset_button.click(fn=reset_game, inputs=[], outputs=[game_log, score_sheet]).then(
        lambda: None, None, None, js=js_scroll
    )

iface.launch()
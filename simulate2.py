import torch
import time
from yahtzee import YahtzeeGame
import utils
import dqn_agent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALL_ACTIONS = utils.generate_all_actions()


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


def format_keep_mask(keep_mask):
    """
    Converts a keep mask (list of bools) into a readable format, e.g., [KK__K].
    """
    return "[" + "".join("K" if keep else "_" for keep in keep_mask) + "]"


def observe_agent(model, env_cls, epsilon=0.0, delay=1.0):
    """
    Observes a trained agent's behavior by playing a single episode of Yahtzee.

    Args:
        model: The trained DQN model.
        env_cls: The environment constructor.
        epsilon: Exploration rate (set to 0 for fully greedy behavior).
        delay: Time delay (in seconds) between steps for better visualization.
    """
    model.eval()  # Set model to evaluation mode
    env = env_cls()
    state_dict = env.reset()
    state = env.get_encoded_state()
    state = torch.FloatTensor(state).to(device)
    done = False
    turn = 0  # Explicit turn counter
    rolls_this_turn = 0  # Track rolls within a turn

    print("\n===== STARTING SIMULATION =====\n")

    while not done:
        if rolls_this_turn == 0:
            turn += 1
            print(f"\n===== TURN {turn} =====")
        
        rolls_this_turn += 1
        print(f"Roll {rolls_this_turn}: Dice = {env.dice}")

        # Get valid actions
        valid_actions_mask = env.get_valid_actions_mask()

        # Select action using trained model
        action_idx = dqn_agent.select_action(model, state, valid_actions_mask, epsilon=epsilon)

        # Convert action index to game action
        if action_idx < 32:  # Reroll action
            keep_mask = [bool(int(bit)) for bit in f"{action_idx:05b}"]
            formatted_mask = format_keep_mask(keep_mask)
            action = ('reroll', keep_mask)
            action_desc = f"Rerolling, keeping: {formatted_mask}"
        else:  # Scoring action
            cat_idx = action_idx - 32
            category = list(env.categories.keys())[cat_idx]
            action = ('score', category)
            action_desc = f"Scoring in category: {category}"

        print(f"Chosen action: {action_desc}")

        # Execute action
        prev_categories = env.categories.copy()  # Store previous scores before action
        next_state, reward, done, _ = env.step(action)
        state = torch.FloatTensor(next_state).to(device)

        # If action was scoring, display the score achieved and reset roll count
        if action[0] == "score":
            scored_value = env.categories[category] if env.categories[category] is not None else 0
            prev_value = prev_categories[category] if prev_categories[category] is not None else 0
            score_earned = scored_value - prev_value  # Difference in score
            print(f"→ Scored {score_earned} points in {category}")
            rolls_this_turn = 0  # Reset roll count for next turn

        # Pause to allow visualization
        time.sleep(delay)

    # Display final score and filled categories
    final_score = sum(v for v in env.categories.values() if v is not None) + env.upper_bonus + env.yahtzee_bonuses

    print("\n===== FINAL GAME STATE =====")
    print(f"Final Score: {final_score}")
    print("Category Scores:")
    for cat, score in env.categories.items():
        print(f"  {cat}: {score}")
    print("Upper Bonus: ", env.upper_bonus)
    print("Yahtzee Bonus: ", env.yahtzee_bonuses)

    print("\n===== END OF OBSERVATION =====")

    return final_score, env.categories



# Define environment to get state size
env = YahtzeeGame()
state_dim = len(env.get_encoded_state())
action_dim = len(ALL_ACTIONS)

# Load trained model
checkpoint_path = "/home/mc5635/yahtzee/yahtzee_rl/saved_models/firm-breeze-148"  # Update with your actual path
trained_model = load_trained_model(checkpoint_path, dqn_agent.DuelingDQN)

# Observe the agent
observe_agent(trained_model, YahtzeeGame, epsilon=0.0, delay=2.0)

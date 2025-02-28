import torch
import numpy as np

CATEGORY_INDEX = {
    'Ones': 0, 'Twos': 1, 'Threes': 2, 'Fours': 3, 'Fives': 4, 'Sixes': 5,
    'Three of a Kind': 6, 'Four of a Kind': 7, 'Full House': 8,
    'Small Straight': 9, 'Large Straight': 10, 'Yahtzee': 11, 'Chance': 12
}

# try one-hot encoding dice

def encode_dice(dice):
    """Encode dice as sorted values and counts."""
    sorted_dice = sorted(dice)
    normalized_dice = torch.tensor(sorted_dice, dtype=torch.float32) / 6.0  # Normalize to [0,1]
    
    dice_counts = torch.zeros(6)
    for d in dice:
        dice_counts[d-1] += 1  # Count occurrences of each face
    
    return torch.cat([dice_counts, normalized_dice])  # Shape: (6 + 5) = (11,)

def encode_rolls_left(rolls_left):
    """Normalize remaining rolls to [0,1]."""
    return torch.tensor([rolls_left / 3.0], dtype=torch.float32)  # Shape: (1,)

def encode_categories(categories):
    """Encode category availability and normalized scores."""
    category_availability = torch.zeros(len(CATEGORY_INDEX))
    category_scores = torch.zeros(len(CATEGORY_INDEX))
    
    for cat, score in categories.items():
        if score is None:
            category_availability[CATEGORY_INDEX[cat]] = 1.0  # Available
        else:
            category_scores[CATEGORY_INDEX[cat]] = score / 50.0  # Normalize score
    
    # return torch.cat([category_availability, category_scores])  # Shape: (13 + 13) = (26,)
    return torch.cat([category_availability])  # Shape: (13 + 13) = (26,)

def encode_upper_sum(categories):
    """sum of upper score sheet"""
    upper_section = list(categories.items())[:6]
    upper_total = sum(v for k, v in upper_section if v is not None)
    normalized_sum = upper_total / 100
    return torch.tensor([normalized_sum], dtype=torch.float32)

def encode_upper_bonus(upper_bonus):
    """Binary flag for upper bonus."""
    return torch.tensor([1.0 if upper_bonus == 35 else 0.0], dtype=torch.float32)  # Shape: (1,)

def encode_yahtzee_bonus(yahtzee_bonuses):
    """Normalize Yahtzee bonuses assuming max of 5 bonuses."""
    return torch.tensor([yahtzee_bonuses / 5.0], dtype=torch.float32)  # Shape: (1,)

def encode_state(game_state):
    """Convert a Yahtzee game state dictionary into a PyTorch tensor."""
    dice_encoded = encode_dice(game_state['dice'])
    rolls_encoded = encode_rolls_left(game_state['rolls_left'])
    categories_encoded = encode_categories(game_state['categories'])
    upper_total_encoded = encode_upper_sum(game_state['categories'])
    upper_bonus_encoded = encode_upper_bonus(game_state['upper_bonus'])
    yahtzee_bonus_encoded = encode_yahtzee_bonus(game_state['yahtzee_bonuses'])
    
    return torch.cat([
        dice_encoded, rolls_encoded, categories_encoded, upper_total_encoded, upper_bonus_encoded, yahtzee_bonus_encoded
    ])  
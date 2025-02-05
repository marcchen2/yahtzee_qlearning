import random
from collections import OrderedDict
import torch

###############################
# Toy Environment Code
###############################

def toy_encode_state(game_state):
    """
    Example minimal encoder for:
      - 3 dice
      - 3 categories: "Ones", "Twos", "Chance"
      - up to 2 rolls each turn (rolls_left in {0,1,2})
    
    Proposed 16 dims:
      0-5:   dice_counts (#1..#6)
      6-8:   sorted_dice / 6.0
      9:     rolls_left / 2.0  (normalized to [0..1])
      10-12: category availability
      13-15: category scores (normalized)
    """
    dice = sorted(game_state['dice'])
    dice_counts = torch.zeros(6)
    for d in game_state['dice']:
        dice_counts[d - 1] += 1
    
    # normalized dice
    sorted_dice = torch.tensor(dice, dtype=torch.float32) / 6.0
    
    # rolls_left / 2 => in [0,1]
    rolls_left = torch.tensor([game_state['rolls_left'] / 2.0], dtype=torch.float32)
    
    # 3 categories
    categories = ['Ones', 'Twos', 'Chance']
    cat_avail = torch.zeros(3)
    cat_score = torch.zeros(3)
    for i, cat in enumerate(categories):
        score_val = game_state['categories'][cat]
        if score_val is None:
            cat_avail[i] = 1.0
        else:
            # e.g. simple normalization by 15
            cat_score[i] = score_val / 15.0
    
    return torch.cat([dice_counts, sorted_dice, rolls_left, cat_avail, cat_score])


class MiniYahtzeeGame:
    """
    A toy environment with:
      - 3 dice
      - 3 categories: "Ones", "Twos", "Chance"
      - up to 2 rolls per turn
      - ends after all 3 categories are used
    """
    def __init__(self):
        self.categories = OrderedDict([
            ('Ones', None),
            ('Twos', None),
            ('Chance', None)
        ])
        self.dice = [0,0,0]
        self.rolls_left = 0  # increment to 2 when a new turn starts

    def reset(self):
        for cat in self.categories:
            self.categories[cat] = None
        self.dice = [0,0,0]
        self.rolls_left = 0
        return self.get_state()

    def get_state(self):
        return {
            'categories': self.categories.copy(),
            'dice': self.dice.copy(),
            'rolls_left': self.rolls_left
        }

    def get_encoded_state(self):
        return toy_encode_state(self.get_state())

    def roll_dice(self, keep_mask=None):
        if keep_mask is None:
            # roll all dice
            self.dice = [random.randint(1,6) for _ in range(3)]
        else:
            for i in range(3):
                if not keep_mask[i]:
                    self.dice[i] = random.randint(1,6)
        self.dice.sort()

    def calculate_score(self, category, dice):
        if category == 'Ones':
            return sum(d for d in dice if d == 1)
        elif category == 'Twos':
            return sum(d for d in dice if d == 2)
        elif category == 'Chance':
            return sum(dice)
        return 0

    def step(self, action):
        """
        action: ('reroll', keep_mask) or ('score', category)
        """
        reward = 0.0
        done = False

        if action[0] == 'reroll':
            keep_mask = action[1]
            # If we are starting a new turn, set rolls_left=2
            if self.rolls_left == 0:
                self.rolls_left = 2
            if self.rolls_left > 0:
                self.roll_dice(keep_mask)
                self.rolls_left -= 1

        elif action[0] == 'score':
            category = action[1]
            # If it's not used
            if self.categories[category] is None:
                sc = self.calculate_score(category, self.dice)
                self.categories[category] = sc
                reward = sc
            # end the turn
            self.rolls_left = 0

            # if all categories are filled -> game ends
            if all(v is not None for v in self.categories.values()):
                done = True
        
        next_state = self.get_state()
        return next_state, reward, done, {}

###############################
# Action Space
###############################

def toy_generate_all_actions():
    """
    e.g. 8 keep masks for 3 dice (2^3) + 1 "roll all" (None) + 3 score categories
    = 8 + 1 + 3 = 12 total
    """
    actions = []
    
    # reroll keep masks
    for mask_int in range(8):  # 2^3
        keep_mask = [(mask_int & (1 << i)) != 0 for i in range(3)]
        actions.append(('reroll', keep_mask))
    # Also a "reroll-all" action (None)
    actions.append(('reroll', None))

    # 3 scoring categories
    for cat in ['Ones', 'Twos', 'Chance']:
        actions.append(('score', cat))

    return actions

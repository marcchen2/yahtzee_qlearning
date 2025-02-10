import random
from collections import OrderedDict
import numpy as np

# If you have a custom state encoder, you can import/use it here.
# from state_rep import encode_state

def encode_state(state_dict):
    """
    Placeholder for your custom encoding logic.
    Returns a simple flattened representation here, just for illustration.
    """
    dice = state_dict['dice']
    categories_used = [1.0 if v is not None else 0.0 for v in state_dict['categories'].values()]
    rolls_left = [float(state_dict['rolls_left'])]
    return np.array(list(dice) + categories_used + rolls_left, dtype=float)

class SimpleYahtzeeGame:
    def __init__(self):
        # Five simple categories
        self.categories = OrderedDict([
            ('Ones', None),
            ('Twos', None),
            ('Threes', None),
            ('Three of a Kind', None),
            ('Chance', None),
        ])
        
        # We'll use 3 dice
        self.dice = [0]*3
        
        # We'll allow 2 rolls per turn
        self.rolls_left = 0

    def reset(self):
        # Reset categories to unused (None)
        for cat in self.categories:
            self.categories[cat] = None
        
        # Reset dice and rolls
        self.dice = [0]*3
        self.rolls_left = 2
        return self.get_encoded_state()

    def get_state(self):
        return {
            'categories': self.categories.copy(),
            'dice': self.dice.copy(),
            'rolls_left': self.rolls_left
        }

    def get_encoded_state(self):
        return encode_state(self.get_state())

    def set_state(self, state):
        self.categories = state['categories'].copy()
        self.dice = state['dice'].copy()
        self.rolls_left = state['rolls_left']

    def roll_dice(self, keep_mask=None):
        """
        Reroll only the dice that are not marked True in keep_mask.
        If keep_mask is None, reroll all dice.
        """
        if keep_mask is None:
            # Generate all 3 dice
            self.dice = np.random.randint(1, 7, size=3)
        else:
            # Generate new values only for non-kept dice
            new_values = np.random.randint(1, 7, size=3)
            self.dice = np.where(keep_mask, self.dice, new_values)
        
        # Sort the dice for a consistent representation
        self.dice.sort()

    def get_possible_moves(self):
        """List of categories still available for scoring."""
        return [cat for cat, score in self.categories.items() if score is None]

    def calculate_score(self, category, dice):
        """
        Return the immediate score you'd get by applying 'category' to the dice.
        """
        dice_np = np.array(dice)
        counts = np.bincount(dice_np, minlength=7)[1:]  # counts[0] is for face '1', etc.
        
        if category == 'Ones':
            return sum(d for d in dice if d == 1)
        elif category == 'Twos':
            return sum(d for d in dice if d == 2)
        elif category == 'Threes':
            return sum(d for d in dice if d == 3)
        elif category == 'Three of a Kind':
            return sum(dice) if max(counts) >= 3 else 0
        elif category == 'Chance':
            return sum(dice)
        else:
            return 0

    def apply_move(self, category):
        """
        Assign (score) that category using current dice, and mark it used in self.categories.
        """
        score = self.calculate_score(category, self.dice)
        self.categories[category] = score
        return score

    def step(self, action):
        """
        action is either:
          ('reroll', keep_mask) -- keep_mask is a list of booleans of length 3
          ('score', category_name)
        Returns: next_state, reward, done, info
        """
        reward = 0.0
        done = False

        if action[0] == 'reroll':
            keep_mask = action[1]
            # If rolls_left == 0, treat this as a new turn (i.e. allow up to 2 rolls).
            if self.rolls_left == 0:
                self.rolls_left = 2
            
            # Perform the reroll if we still have rolls left
            if self.rolls_left > 0:
                self.roll_dice(keep_mask)
                self.rolls_left -= 1

        elif action[0] == 'score':
            category = action[1]
            # If category is still unused, apply move
            if self.categories[category] is None:
                reward = self.apply_move(category)
            else:
                reward = 0  # Already used category => no additional score

            # After scoring, your turn ends (rolls_left = 0)
            self.rolls_left = 0

            # Check if the game is over (all categories filled)
            if all(v is not None for v in self.categories.values()):
                done = True

        next_state = self.get_encoded_state()
        return next_state, reward, done, {}

    def get_valid_actions_mask(self):
        """
        Returns a boolean mask for all possible actions.
        With 3 dice, there are 2^3 = 8 keep-mask patterns for rerolls,
        plus 5 categories = total 13 possible discrete actions.
        
        Indices 0..7 => reroll patterns
        Indices 8..12 => scoring categories
        """
        mask = [False]*13
        
        # If we can still roll, enable the 8 reroll actions
        if self.rolls_left > 0:
            for i in range(8):
                mask[i] = True
        
        # If we cannot roll anymore, enable any category that's unused
        else:
            cat_list = list(self.categories.keys())
            for i, cat in enumerate(cat_list):
                if self.categories[cat] is None:
                    mask[8 + i] = True
        
        return mask


# A simple random agent for demonstration
class DumbAgent:
    def choose_dice_to_keep(self, current_dice, rolls_left):
        # Choose keep_mask randomly: for each die, True (keep) or False (reroll)
        return [bool(random.getrandbits(1)) for _ in current_dice]

    def choose_category(self, game_state):
        available_cats = [cat for cat, score in game_state['categories'].items() if score is None]
        return random.choice(available_cats)

def simulation_mode(agent, num_games=1):
    """
    Runs a few complete games with the DumbAgent to illustrate how the environment works.
    """
    for g in range(num_games):
        game = SimpleYahtzeeGame()
        state = game.reset()
        turn_counter = 1
        
        # Game ends once all categories are used
        while None in game.categories.values():
            print(f"\n=== Turn {turn_counter} ===")

            # Up to 2 rolls per turn
            for roll_index in range(2):
                if roll_index == 0:
                    # First roll => reroll all dice
                    state, reward, done, _ = game.step(("reroll", None))
                    print(f"  Roll #1: {game.dice}")
                else:
                    # Second roll => choose which dice to keep
                    keep_mask = agent.choose_dice_to_keep(game.dice, game.rolls_left)
                    keep_str = "".join("K" if k else "_" for k in keep_mask)

                    state, reward, done, _ = game.step(("reroll", keep_mask))
                    print(f"  Roll #2: {game.dice} (mask: {keep_str})")

                if game.rolls_left <= 0:
                    break

            # Score a category
            category = agent.choose_category(game.get_state())
            state, score_gained, done, _ = game.step(("score", category))

            total_score = sum(v for v in game.categories.values() if v is not None)
            print(f"  Final dice: {game.dice}")
            print(f"  Chose category: {category}, got {score_gained} points")
            print(f"  Current total score: {total_score}")

            turn_counter += 1
            if done:
                break

        final_score = sum(v for v in game.categories.values() if v is not None)
        print(f"\nGame Over! Final Score: {final_score}\n")


if __name__ == "__main__":
    agent = DumbAgent()
    
    print("\nRunning a quick simulation with the DumbAgent...")
    simulation_mode(agent, num_games=1)

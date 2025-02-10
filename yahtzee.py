import random
from collections import OrderedDict
import numpy as np
from state_rep import encode_state

import random
from collections import OrderedDict
import numpy as np
from state_rep import encode_state

class YahtzeeGame:
    def __init__(self):
        self.categories = OrderedDict([
            ('Ones', None), ('Twos', None), ('Threes', None), ('Fours', None),
            ('Fives', None), ('Sixes', None), ('Three of a Kind', None),
            ('Four of a Kind', None), ('Full House', None),
            ('Small Straight', None), ('Large Straight', None),
            ('Yahtzee', None), ('Chance', None)
        ])
        self.upper_bonus = 0
        self.yahtzee_bonuses = 0
        self.dice = [0]*5
        # Start with 0 rolls_left; it will be set to 3 internally 
        # whenever the first reroll is requested each turn:
        self.rolls_left = 0

    def reset(self):
        for cat in self.categories:
            self.categories[cat] = None
        self.upper_bonus = 0
        self.yahtzee_bonuses = 0
        self.dice = [0]*5
        self.rolls_left = 3
        return self.get_encoded_state()

    def get_state(self):
        return {
            'categories': self.categories.copy(),
            'dice': self.dice.copy(),
            'rolls_left': self.rolls_left,
            'upper_bonus': self.upper_bonus,
            'yahtzee_bonuses': self.yahtzee_bonuses
        }
    
    def get_encoded_state(self):
        return encode_state(self.get_state())

    def set_state(self, state):
        self.categories = state['categories'].copy()
        self.dice = state['dice'].copy()
        self.rolls_left = state['rolls_left']
        self.upper_bonus = state['upper_bonus']
        self.yahtzee_bonuses = state['yahtzee_bonuses']

    def roll_dice(self, keep_mask=None):
        if keep_mask is None:
            # Generate all 5 dice at once
            self.dice = np.random.randint(1, 7, size=5)
        else:
            # Generate new values only for non-kept dice
            new_values = np.random.randint(1, 7, size=5)
            self.dice = np.where(keep_mask, self.dice, new_values)
        self.dice.sort()

    def get_possible_moves(self):
        return [cat for cat, score in self.categories.items() if score is None]

    def calculate_score(self, category, dice):
        dice_np = np.array(dice)
        counts = np.bincount(dice_np, minlength=7)[1:]  # Count occurrences of 1-6
        number_map = {
            'Ones':   1,
            'Twos':   2,
            'Threes': 3,
            'Fours':  4,
            'Fives':  5,
            'Sixes':  6
        }
        if category in number_map:
            value = number_map[category]
            return sum(d for d in dice if d == value)
        elif category == 'Three of a Kind':
            return sum(dice) if max(counts) >= 3 else 0
        elif category == 'Four of a Kind':
            return sum(dice) if max(counts) >= 4 else 0
        elif category == 'Full House':
            return 25 if (3 in counts and 2 in counts) else 0
        elif category == 'Small Straight':
            return 30 if any(all(x in dice for x in [i,i+1,i+2,i+3]) for i in [1,2,3]) else 0
        elif category == 'Large Straight':
            return 40 if any(all(x in dice for x in [i,i+1,i+2,i+3,i+4]) for i in [1,2]) else 0
        elif category == 'Yahtzee':
            return 50 if all(d == dice[0] for d in dice) else 0
        elif category == 'Chance':
            return sum(dice)
        return 0

    def apply_move(self, category):
        
        score = self.calculate_score(category, self.dice)
        bonuses = 0
        
        # Check if we're crossing 63 in the upper section for the first time
        if category in ['Ones', 'Twos', 'Threes', 'Fours', 'Fives', 'Sixes']:
            upper_section = list(self.categories.items())[:6]
            upper_total = sum(v for k, v in upper_section if v is not None)
            if upper_total + score >= 63 and self.upper_bonus == 0:
                self.upper_bonus = 35
                bonuses += 35

        # Extra Yahtzee bonus if we've already scored Yahtzee category previously
        if category == 'Yahtzee' and score == 50 and self.categories['Yahtzee'] is not None:
            self.yahtzee_bonuses += 100
            bonuses += 100

        self.categories[category] = score
        return score, bonuses

    def step(self, action):
        """
        action is either:
          ('reroll', keep_mask)
          ('score', category_name)
        """
        reward = 0.0
        done = False

        if action[0] == 'reroll':
            keep_mask = action[1]
            # If rolls_left == 0, treat this as the start of a brand-new turn
            # (i.e. allow up to 3 rolls this turn).
            if self.rolls_left == 0:
                self.rolls_left = 3

            if self.rolls_left > 0:
                self.roll_dice(keep_mask)
                self.rolls_left -= 1  # use up one roll

        elif action[0] == 'score':
            category = action[1]
            if self.categories[category] is not None:
                # Category used already
                reward = 0
            else:
                score, bonuses = self.apply_move(category)
                reward = score + bonuses

            # Scoring ends the turn; set rolls_left to 3
            self.rolls_left = 3

            # Check if game ended (all categories filled)
            if all(v is not None for v in self.categories.values()):
                done = True

        next_state = self.get_encoded_state()
        return next_state, reward, done, {}

    def get_valid_actions_mask(self):
        """Returns boolean mask for all 45 actions (32 reroll patterns + 13 categories)"""
        mask = [False] * 45
        
        # if at the beginning of a turn, must roll all dice. 
        if self.rolls_left == 3:
            mask[0] = True
            
        elif self.rolls_left > 0:
            # Rolling phase: can reroll (first 32 actions)
            mask[:32] = [True] * 32
        else:
            # Scoring phase: can choose unused categories
            available_cats = [i for i, (cat, score) in enumerate(self.categories.items()) 
                            if score is None]
            for idx in available_cats:
                mask[32 + idx] = True  # Scoring actions start at index 32
                
        return mask


class DumbAgent:
    def choose_dice_to_keep(self, current_dice, rolls_left):
        return [bool(random.getrandbits(1)) for _ in range(5)]  # random T/F

    def choose_category(self, game_state):
        available = [cat for cat, score in game_state['categories'].items() if score is None]
        return random.choice(available)

def simulation_mode(agent, num_games=1):
    for _ in range(num_games):
        game = YahtzeeGame()
        state = game.reset()
        turn_counter = 1
        
        while None in game.categories.values():
            print(f"\n=== Turn {turn_counter} ===")

            # 3 total rolls
            # game.rolls_left = 3

            # Up to 3 rolls
            for roll_index in range(3):
                
                if roll_index == 0:
                    # First roll: always reroll all dice
                    state, reward, done, _ = game.step(("reroll", None))
                    print(f"  Roll #1 (sorted): {game.dice}")
                else:
                    # Subsequent rolls: choose dice to keep
                    keep_mask = agent.choose_dice_to_keep(game.dice, game.rolls_left)
                    keep_str = "".join("K" if k else "_" for k in keep_mask)

                    state, reward, done, _ = game.step(("reroll", keep_mask))
                    print(f"  Roll #{roll_index+1} (sorted): {game.dice}  (mask: {keep_str})")
                
                if game.rolls_left <= 0:
                    break

            # Score a category
            category = agent.choose_category(game.get_state())
            prev_bonus = game.upper_bonus + game.yahtzee_bonuses
            state, score_gained, done, _ = game.step(("score", category))

            category_scores = sum(v for v in game.categories.values() if v is not None)
            total_score = category_scores + game.upper_bonus + game.yahtzee_bonuses

            print(f"  Final dice (sorted): {game.dice}")
            print(f"  Chose category: {category}, got {score_gained} points")
            if (game.upper_bonus + game.yahtzee_bonuses) > prev_bonus:
                print(f"  Bonus awarded! (Upper/Yahtzee bonuses now: "
                      f"{game.upper_bonus} / {game.yahtzee_bonuses})")
            print(f"  Current total score: {total_score}")

            turn_counter += 1
            if done:
                break

        final_score = sum(v for v in game.categories.values() if v is not None)
        final_score += game.upper_bonus + game.yahtzee_bonuses
        print(f"\nGame Over! Final Score: {final_score}\n")



import itertools

def all_possible_keep_patterns(dice):
    """Generate all unique value-based keep patterns for current dice"""
    patterns = set()
    counts = {num: dice.count(num) for num in set(dice)}
    
    # Generate possible value combinations
    for num in counts:
        for keep_count in range(counts[num] + 1):
            if keep_count == 0:
                continue
            # Generate mask for this value count
            pattern = []
            kept = 0
            for d in dice:
                if d == num and kept < keep_count:
                    pattern.append(True)
                    kept += 1
                else:
                    pattern.append(False)
            patterns.add(tuple(pattern))
    
    # Add combinations with multiple numbers
    for combo in itertools.combinations(set(dice), 2):
        for count1 in range(1, dice.count(combo[0]) + 1):
            for count2 in range(1, dice.count(combo[1]) + 1):
                pattern = []
                kept1 = kept2 = 0
                for d in dice:
                    if d == combo[0] and kept1 < count1:
                        pattern.append(True)
                        kept1 += 1
                    elif d == combo[1] and kept2 < count2:
                        pattern.append(True)
                        kept2 += 1
                    else:
                        pattern.append(False)
                patterns.add(tuple(pattern))
    
    return [list(p) for p in patterns]

def calculation_mode(agent):
    game = YahtzeeGame()
    
    # Get input with validation
    try:
        dice = list(map(int, input("Enter dice values (space-separated, 5 numbers): ").split()))
        assert len(dice) == 5, "Must enter exactly 5 dice values"
        used = input("Enter used categories (comma-separated): ").split(',')
        rolls_left = int(input("Enter rolls remaining (0-2): "))
        turn = int(input("Enter current turn (1-13): "))
    except Exception as e:
        print(f"Invalid input: {e}")
        return

    # Set game state
    game.dice = dice
    game.rolls_left = rolls_left
    for cat in used:
        if cat.strip():
            game.categories[cat.strip()] = 0
    
    # Get predictions
    print(f"\nAnalysis for Turn {turn} with {rolls_left} rolls remaining:")
    available = game.get_possible_moves()
    
    # Get keep patterns
    print("\nDice Keeping Recommendations:")
    patterns = all_possible_keep_patterns(dice)
    for pattern in patterns:
        keep_str = "".join(["K" if k else "_" for k in pattern])
        kept = [d for d, k in zip(dice, pattern) if k]
        
        # Calculate value-based pattern identifier
        value_counts = {}
        for num in kept:
            value_counts[num] = value_counts.get(num, 0) + 1
        pattern_id = "+".join(f"{v}x{k}" for k,v in sorted(value_counts.items()))
        
        # Placeholder evaluation score
        eval_score = random.uniform(0, 50)
        print(f"{keep_str} ({pattern_id}): {eval_score:.1f}")
    
    # Show category predictions
    print("\nCategory Predictions:")
    for move in available:
        q_value = random.uniform(0, 50)  # Placeholder
        immediate = game.calculate_score(move, dice)
        upper_bonus_impact = ""
        
        if move in ['Ones', 'Twos', 'Threes', 'Fours', 'Fives', 'Sixes']:
            current_upper = sum(v for k,v in list(game.categories.items())[:6] if v is not None)
            potential_upper = current_upper + immediate
            upper_bonus_impact = f" (Upper Bonus: {'+' if potential_upper >=63 else '-'})"
            
        print(f"{move:<20} | Q: {q_value:5.1f} | Now: {immediate:3d}{upper_bonus_impact}")
        
def performance_mode(agent, num_games=100):
    scores = []
    for _ in range(num_games):
        game = YahtzeeGame()
        game.reset()

        # 13 turns total
        for _ in range(13):
            # Set rolls_left = 3 for each new turn (same as simulation_mode).
            # game.rolls_left = 3

            # Perform up to 3 rolls this turn
            for roll_index in range(3):
                # if game.rolls_left <= 0:
                #     break
                if roll_index == 0:
                    # First roll: reroll all dice
                    _, _, done, _ = game.step(("reroll", None))
                else:
                    # Subsequent rolls: choose which dice to keep
                    keep_mask = agent.choose_dice_to_keep(game.dice, game.rolls_left)
                    _, _, done, _ = game.step(("reroll", keep_mask))

            # Now choose a category to score (ending the turn)
            available = game.get_possible_moves()
            if not available:  # If no categories left, game is essentially done
                break
            category = agent.choose_category(game.get_state())
            _, _, done, _ = game.step(("score", category))

            if done:
                # If the game is flagged done (all categories filled), break early
                break

        total_score = sum(v for v in game.categories.values() if v is not None)
        total_score += game.upper_bonus + game.yahtzee_bonuses
        scores.append(total_score)

    print(f"\nPerformance over {num_games} games:")
    print(f"Mean score: {np.mean(scores):.1f}")
    print(f"Median score: {np.median(scores):.1f}")
    
    


if __name__ == "__main__":
    agent = DumbAgent()
    
    # while True:
    print("\nSelect mode:")
    print("1. Simulation Mode")
    print("2. Calculation Mode")
    print("3. Performance Mode")
    print("4. Exit")
    choice = input("Enter choice: ")
    
    if choice == '1':
        simulation_mode(agent)
    elif choice == '2':
        calculation_mode(agent)
    elif choice == '3':
        performance_mode(agent)
    # elif choice == '4':
    #     break
    else:
        print("Invalid choice")
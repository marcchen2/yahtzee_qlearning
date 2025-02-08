import itertools

YAHTZEE_CATEGORIES = [
    'Ones', 'Twos', 'Threes', 'Fours', 'Fives', 'Sixes',
    'Three of a Kind', 'Four of a Kind', 'Full House',
    'Small Straight', 'Large Straight', 'Yahtzee', 'Chance'
]

def generate_all_actions():
    actions = []
    
    # 1) All possible keep masks: 2^5 = 32
    for mask_int in range(32):
        keep_mask = [(mask_int & (1 << i)) != 0 for i in range(5)]
        # keep_mask is a list of T/F for each die
        actions.append(('reroll', keep_mask))
    
    # 2) All possible scoring categories (13)
    for cat in YAHTZEE_CATEGORIES:
        actions.append(('score', cat))

    return actions

ALL_ACTIONS = generate_all_actions()  # length = 32 + 13 = 45

if __name__ == "__main__":
    ALL_ACTIONS = generate_all_actions()  # length = 32 + 13 = 45
    print(ALL_ACTIONS)
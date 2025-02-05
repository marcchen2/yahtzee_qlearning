import utils
from yahtzee import YahtzeeGame
import numpy as np
import random 


ALL_ACTIONS = utils.generate_all_actions()

def random_agent(env_cls, num_episodes=100):
    scores = []
    for _ in range(num_episodes):
        env = env_cls()
        env.reset()
        done = False
        while not done:
            action = random.choice(ALL_ACTIONS)
            _, _, done, _ = env.step(action)
        # e.g. final score from env
        final_score = sum(v for v in env.categories.values() if v is not None)
        final_score += env.upper_bonus + env.yahtzee_bonuses
        scores.append(final_score)
    return np.mean(scores)

print(f"Random agent average score: {random_agent(YahtzeeGame, 100)}")
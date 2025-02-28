# Yahtzee Reinforcement Q-Learning Agent

Yahtzee is a dice game where the goal is to roll high-scoring combinations of dice. The player must balance risk and reward while adapting to the randomness of the dice. The optimal Yahtzee strategy has actually been discovered, and can be obtained by [computing the expected value of every possible game state](http://www.yahtzee.org.uk/optimal_yahtzee_TV.pdf "http://www.yahtzee.org.uk/optimal_yahtzee_TV.pdf by Tom Verhoeff"), and scores a median of 248. This project instead implements a Deep Q-Network (DQN) agent to see if it can approach optimal play with a model-free approach. 

The agent uses a dueling network architecture to separate state value estimation from action advantage estimation, allowing for more efficient learning. My best model so far achieves a median score of 211 over 1000 games. Compare this to a baseline of ~45 for random valid actions, and ~204 achieved in the 2018 paper [Reinforcement Learning for Solving Yahtzee](https://web.stanford.edu/class/aa228/reports/2018/final75.pdf "PDF of the paper by Minhyung Kang and Luca Schroeder"). 


## Architecture 

- **Dueling DQN**: Separate state value and action advantage estimation
- **Prioritized Experience Replay**: Replay important experiences more frequently to speed up learning
- **Double DQN**: Reduce overestimation bias by using two networks for Q-value evaluation
- **Action Masking**: Only allow valid actions to be taken by setting invalid actions' q-values to `-Inf`.
- **Reward Shaping**: Reward the agent for scoring points in categories and bonuses on the turns they are scored, as well as the final score of the episode at the end. The hope is that the intermediate rewards make the rewards less sparse to learn a somewhat effective strategy, and that the final score reward allows it to optimize its overall decision making. 
- **State Representation**: Dice are sorted and represented as both counts and normalized values. Categories are represented only as scored or unscored, but the sum of the upper section is also included. Rolls left in the turn are encoded, and the upper and yahtzee bonuses are also included. 

## Next Steps

I want to try sorted 1-hot encodings of the dice and adding some light probabilistic heuristics to the state space, similar to what a human would do in terms of the evaluating the probabilities of scoring a category given the current dice. This second idea would venture into a hybrid model-based/model-free approach. 



## To Run Demo:

The Gradio UI allows you to observe the agent play, analyze its decisions, and evaluate its performance. 

`git clone https://github.com/marcchen2/yahtzee_qlearning.git`

`pip install -r requirements`

`python3 main_ui.py`


# Key Resources:


Kang, M., & Schroeder, L. (2018). Reinforcement Learning for Solving Yahtzee. Stanford University. Available at https://web.stanford.edu/class/aa228/reports/2018/final75.pdf

Wang, Z., Schaul, T., Hessel, M., van Hasselt, H., Lanctot, M., & de Freitas, N. (2016). Dueling Network Architectures for Deep Reinforcement Learning. arXiv preprint arXiv:1511.06581. Available at https://arxiv.org/pdf/1511.06581

Hessel, M., Modayil, J., van Hasselt, H., Schaul, T., Ostrovski, G., Dabney, W., Horgan, D., Piot, B., Azar, M., & Silver, D. (2017). Rainbow: Combining Improvements in Deep Reinforcement Learning. arXiv preprint arXiv:1710.02298. Available at https://arxiv.org/pdf/1710.02298

Brunton, S. (2022, January 15). Q-Learning: Model Free Reinforcement Learning and Temporal Difference Learning [Video]. YouTube. https://www.youtube.com/watch?v=0iqz4tcKN58

Used Cursor and/or DeepSeek to: 
- Create the baseline code for the Yahtzee environment. 
- Debug the training pipeline.
- Help me write the Gradio UI elements. 

Weights and Biases to track performance of Models


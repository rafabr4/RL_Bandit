from os import system

import numpy as np
import sys

from environments.bandit import BanditEnv
from simulation.experiments import BanditExperiment
from animator import All_Plots_Animation

class EpsilonGreedy():
    """
    Plays in an Epsilon-Greedy fasion
    """
    def __init__(self, num_actions, epsilon):
        self.num_actions = num_actions
        self.name = "Epsilon Greedy"
        # Initialize rewards and counts to 0
        self.total_rewards = np.zeros(num_actions, dtype = np.longdouble)
        self._total_counts = np.zeros(num_actions, dtype = np.longdouble)
        if (epsilon is None or epsilon < 0 or epsilon > 1):
            print("EpsilonGreedy: Invalid value of epsilon")
            sys.exit(0)
        else:
            self.epsilon = epsilon
    
    def act(self):
        """
        Chooses a random action with probability epsilon, or best action
        with probability 1-epsilon.
        """

        choice = None
        # Use a bernoulli trial to decide if playing randomly or greedy
        choice = np.random.binomial(1, self.epsilon)
            
        if choice == 1:
            # If bernoulli trial was 1 (with probability epsilon), play random
            return np.random.choice(self.num_actions)
        else:
            # If bernoulli trial was 0 (with probability 1-epsilon), play best observed action
            current_averages = np.divide(self.total_rewards, self.total_counts, where = self.total_counts > 0)
            # Assign 0.5 "observed" reward for actions that haven't been played, for decision purpose only
            current_averages[self.total_counts <= 0] = 0.5
            current_action = np.argmax(current_averages)
            return current_action
        
    def feedback(self, action, reward):
        """
        Update the rewards and counts of each action according to
        feedback of the environment.
        """
        self.total_rewards[action] += reward
        self.total_counts[action] += 1
    
    @property
    def total_counts(self):
        return self._total_counts

system('cls') # Clear window

# Simulation parameters
evaluation_seed = 3 # Use None for random
num_actions = 20
trials = 50
epsilon = 0.15
distribution = "bernoulli"
fps = 15  # For rendering animation

env = BanditEnv(num_actions, distribution, evaluation_seed)
agent = EpsilonGreedy(num_actions, epsilon)
experiment = BanditExperiment(env, agent)
experiment.run_bandit(max_number_of_trials=trials)

# Prepare data for animations
data = {}
data["algorithm"] = agent.name
data["details"] = {"epsilon": epsilon}
data["total_counts"] = experiment.action_history
data["regret_history"] = experiment.regret_history
data["reward_parameters"] = experiment.env.reward_parameters
data["average_rewards_history"] = experiment.average_rewards
all_plots_animation = All_Plots_Animation(data, fps=fps)

all_plots_animation.generate_animation(render_action="show")
# all_plots_animation.generate_animation(render_action="save")

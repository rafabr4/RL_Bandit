from os import system

import numpy as np
import sys

from environments.bandit import BanditEnv
from simulation.experiments import BanditExperiment
from animator import All_Plots_Animation

class Greedy():
    """
    Plays in an Greedy fasion
    """
    def __init__(self, num_actions):
        self.name = "Greedy"
        # Initialize rewards and counts to 0
        self.total_rewards = np.zeros(num_actions, dtype = np.longdouble)
        self._total_counts = np.zeros(num_actions, dtype = np.longdouble)
    
    def act(self):
        """
        Chooses action with best observed reward
        """
        current_averages = np.divide(self.total_rewards, self.total_counts, where = self.total_counts > 0)
        # Assign 0.5 "observed" reward for actions that haven't been played, for decision purpose only
        current_averages[self._total_counts <= 0] = 0.5
        current_action = np.argmax(current_averages)
        return current_action
        
    def feedback(self, action, reward):
        self.total_rewards[action] += reward
        self.total_counts[action] += 1
    
    @property
    def total_counts(self):
        return self._total_counts

class OptimisticGreedy(Greedy):
    """
    Plays in an Optimistic Greedy fashion
    """
    def __init__(self, num_actions, initial_value):
        Greedy.__init__(self, num_actions)
        self.name = "Optimistic Greedy"
        
        # Rewards are initialized different to initial_value
        self.total_rewards += np.ones(num_actions, dtype=np.longdouble) * initial_value
        self._total_counts += np.ones(num_actions)

system('cls') # Clear window

evaluation_seed = 3 # Use None for random
num_actions = 20
trials = 50
R = 3
distribution = "bernoulli"
fps = 15  # For rendering animation

env = BanditEnv(num_actions, distribution, evaluation_seed)
agent = OptimisticGreedy(num_actions, R)
experiment = BanditExperiment(env, agent)
experiment.run_bandit(max_number_of_trials=trials)

data = {}
data["algorithm"] = agent.name
data["details"] = {"R": R}
data["total_counts"] = experiment.action_history
data["regret_history"] = experiment.regret_history
data["reward_parameters"] = experiment.env.reward_parameters
data["average_rewards_history"] = experiment.average_rewards
all_plots_animation = All_Plots_Animation(data, fps=fps)

all_plots_animation.generate_animation(render_action="show")
# all_plots_animation.generate_animation(render_action="save")

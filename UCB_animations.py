from os import system

import numpy as np
import sys

from environments.bandit import BanditEnv
from simulation.experiments import BanditExperiment
from animator import All_Plots_Animation

class UCB():
    """
    Plays in an Upper Confidence Bound fashion
    """
    def __init__(self, num_actions, C, max_number_of_trials):
        self.num_actions = num_actions
        self.name = "UCB"
        self.round = 0
        self.C = C
        # Initialize rewards, counts and bonuses to 0
        self.total_rewards = np.zeros(num_actions, dtype = np.longdouble)
        self._total_counts = np.zeros(num_actions, dtype = np.longdouble)
        self._bonuses_history = np.zeros((max_number_of_trials, num_actions))
        
    def act(self):
        current_action = None
        self.round += 1
        if self.round <= self.num_actions:
            # Play each action once in the first k rounds (k is the of num_actions)
            current_action = self.round - 1
        else:
            # At round t, play the arms with maximum average plus exploration bonus
            current_averages = np.divide(self.total_rewards, self.total_counts)
            self._bonuses_history[self.round - 1, :] = self.C * np.sqrt(2 * np.log(self.round) / self.total_counts)
            estimates_plus_bonuses = current_averages + self._bonuses_history[self.round - 1, :]
            current_action = np.argmax(estimates_plus_bonuses)

        return current_action
    
    def feedback(self, action, reward):
        self.total_rewards[action] += reward
        self.total_counts[action] += 1
    
    @property
    def total_counts(self):
        return self._total_counts

    @property
    def bonuses_history(self):
        return self._bonuses_history

system('cls') # Clear window

# Simulation parameters
evaluation_seed = 3 # Use None for random
num_actions = 20
trials = 500
C = 0.5
distribution = "bernoulli"
fps = 15  # For rendering animation

env = BanditEnv(num_actions, distribution, evaluation_seed)
agent = UCB(num_actions, C, trials)
experiment = BanditExperiment(env, agent)
experiment.run_bandit(max_number_of_trials=trials)

data = {}
data["algorithm"] = agent.name
data["details"] = {"C": C}
data["total_counts"] = experiment.action_history
data["regret_history"] = experiment.regret_history
data["reward_parameters"] = experiment.env.reward_parameters
data["average_rewards_history"] = experiment.average_rewards
data["bonus_history"] = experiment.agent.bonuses_history
all_plots_animation = All_Plots_Animation(data, fps=fps)

all_plots_animation.generate_animation(render_action="show")
# all_plots_animation.generate_animation(render_action="save")

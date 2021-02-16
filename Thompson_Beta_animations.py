from os import system

import numpy as np
import sys

from environments.bandit import BanditEnv
from simulation.experiments import BanditExperiment
from animator import All_Plots_Animation

class ThompsonBeta():
    """
    Plays in a Thompson Sampling fashion
    The distribution must be Bernoulli because a Beta is used
    """
    def __init__(self, num_actions, max_number_of_trials):
        self.total_rewards = np.zeros(num_actions, dtype = np.longdouble)
        self._total_counts = np.zeros(num_actions, dtype=np.longdouble)
        self.name = "Thompson Beta"
        
        # For each arm, maintain success and failures
        # Start with 1 always
        self._successes_history = np.ones((max_number_of_trials,num_actions), dtype=np.int)
        self._failures_history = np.ones((max_number_of_trials,num_actions), dtype=np.int)
        
        self.round = 0
        
    def act(self):
        prior_sampling = np.random.beta(self.successes_history[self.round,:], self.failures_history[self.round,:])
        
        current_action = np.argmax(prior_sampling)

        return current_action
    
    def feedback(self, action, reward):
        if reward > 0:
            self._successes_history[self.round:, action] += 1
        else:
            self._failures_history[self.round:, action] += 1
        self.total_rewards[action] += reward
        self._total_counts[action] += 1
        self.round += 1
    
    @property
    def total_counts(self):
        return self._total_counts
    
    @property
    def successes_history(self):
        return self._successes_history
    
    @property
    def failures_history(self):
        return self._failures_history

system('cls') # Clear window

# Simulation parameters
evaluation_seed = 3 # Use None for random
num_actions = 20
trials = 500
distribution = "bernoulli"
fps = 15  # For rendering animation

env = BanditEnv(num_actions, distribution, evaluation_seed)
agent = ThompsonBeta(num_actions, trials)
experiment = BanditExperiment(env, agent)
experiment.run_bandit(max_number_of_trials=trials)

data = {}
data["algorithm"] = agent.name
data["details"] = ""
data["total_counts"] = experiment.action_history
data["regret_history"] = experiment.regret_history
data["reward_parameters"] = experiment.env.reward_parameters
data["average_rewards_history"] = experiment.average_rewards
data["successes_history"] = experiment.agent.successes_history
data["failures_history"] = experiment.agent.failures_history
all_plots_animation = All_Plots_Animation(data, fps=fps)

all_plots_animation.generate_animation(render_action="show")
# all_plots_animation.generate_animation(render_action="save")

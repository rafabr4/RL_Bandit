from os import system

import numpy as np
import sys

from environments.bandit import BanditEnv
from simulation.experiments import BanditExperiment
from animator import All_Plots_Animation

class RoundRobin():
    """
    Plays in a Round Robin fashion for some rounds
    Then plays the best observed action
    """

    def __init__(self, num_actions, round_robin_rounds):
        self.num_actions = num_actions
        self.name = "Round Robin"
        self.total_rewards = np.zeros(num_actions, dtype=np.longdouble)
        self._total_counts = np.zeros(num_actions, dtype = np.longdouble)
        self.previous_action = None #keep track of previous action
        self.round_robin_rounds = round_robin_rounds
        self.num_of_rounds = 0
    
    def act(self):
        # Play round robin for a limited number of rounds
        if self.num_of_rounds < self.round_robin_rounds:
        
            # Initialize to 0 if no previous action
            if self.previous_action == None:
                current_action = 0
            else:
                current_action = self.previous_action + 1

            # Reset to 0 if reached the last action
            if current_action == self.num_actions:
                current_action = 0
            
            self.previous_action = current_action
        
        # Then lock on to the best action according to our estimates
        else:
            current_action = np.argmax(self.total_rewards)
        
        self.num_of_rounds += 1

        return current_action
        
    def feedback(self, action, reward):
        self.total_rewards[action] += reward
        self._total_counts[action] += 1
    
    @property
    def total_counts(self):
        return self._total_counts

system('cls') # Clear window

# Simulation parameters
evaluation_seed = 3 # Use None for random
num_actions = 20
trials = 50
max_round_robin_rounds = 10
distribution = "bernoulli"
fps = 15  # For rendering animation

env = BanditEnv(num_actions, distribution, evaluation_seed)
agent = RoundRobin(num_actions, max_round_robin_rounds*num_actions)
experiment = BanditExperiment(env, agent)
experiment.run_bandit(max_number_of_trials=trials)

data = {}
data["algorithm"] = agent.name
data["details"] = {"nrr_rounds": max_round_robin_rounds}
data["total_counts"] = experiment.action_history
data["regret_history"] = experiment.regret_history
data["reward_parameters"] = experiment.env.reward_parameters
data["average_rewards_history"] = experiment.average_rewards
all_plots_animation = All_Plots_Animation(data, fps=fps)

all_plots_animation.generate_animation(render_action="show")
# all_plots_animation.generate_animation(render_action="save")
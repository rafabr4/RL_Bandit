import numpy as np
import sys

class BanditExperiment(object):
    """
    Runs a simulation of the Bandit environment using an agent that
    provides actions, and receives rewards.
    Currently supported agents:
        - Greedy
        - Epsilon Greedy
        - Optimistic Greedy
        - UCB
        - Thompson Beta
    """

    def __init__(self, env, agent):

        self.env = env
        self.agent = agent
        
        self.algorithm_name = agent.name

        self._action_history = None
        self._regret_history = None
        self._average_rewards = None
  
    def run_bandit(self, max_number_of_trials=1000):

        self._action_history = np.zeros((max_number_of_trials, len(self.agent.total_counts)))
        self._regret_history = np.zeros((max_number_of_trials))
        self._average_rewards = np.zeros((max_number_of_trials, len(self.agent.total_counts)))
        
        # For Optimistic Greedy, the initial rewards are different from zero
        if self.algorithm_name == "Optimistic Greedy":
            self._average_rewards += self.agent.total_rewards
        
        print("Distribution:", self.env.distribution, self.env.reward_parameters)
        print("Optimal arm:", self.env.optimal_arm)
        
        cumulative_reward = 0.0 # Only for printing average reward at the end
        cumulative_regret = 0.0
        
        for trial in range(max_number_of_trials):
            action = self.agent.act()
            reward = self.env.step(action)

            cumulative_reward += reward
            
            self.agent.feedback(action, reward)
            
            # Update our history variables
            self._action_history[trial:, action] += 1
            self._average_rewards[trial:, action] = self.agent.total_rewards[action] / self.agent.total_counts[action]
            
            gap = self.env.compute_gap(action) # Gap to best action for updating regret
            if action != self.env.optimal_arm:
                cumulative_regret += gap
            self._regret_history[trial] = cumulative_regret

        print("--------------------------------------------------")
        print("Policy:", self.agent.name, "\nAverage Reward:", cumulative_reward / max_number_of_trials, \
                "\nAverage Regret:", cumulative_regret / max_number_of_trials)
        print("Arm pulls:", self.agent.total_counts)

    @property
    def action_history(self):
        return self._action_history
    
    @property
    def regret_history(self):
        return self._regret_history
    
    @property
    def average_rewards(self):
        return self._average_rewards

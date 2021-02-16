import numpy as np
import sys
        
class BanditEnv():
    """
    Builds an evironment for the K-Armed Bandit problem.
        - num_actions: number of possible actions (arms) in the environment
        - distribution: the nature of the rewards. Possible values are:
            * bernoulli
            * normal
            * heavy-tail
        - evaluation_seed: random generator seed
    The probability of getting a reward is seen as the reward itself.
    """
    
    def __init__(self, num_actions = 10, distribution = "bernoulli", evaluation_seed=None):
        
        self.num_actions = num_actions
        self.distribution = distribution
        
        np.random.seed(evaluation_seed)
        
        # Define true reward parameters according to distribution chosen
        self.reward_parameters = None
        if distribution == "bernoulli":
            self.reward_parameters = np.random.rand(num_actions)
        elif distribution == "normal":
            # Define reward according to a mean value taken from normal distribution, and a noise value in [0,1)
            self.reward_parameters = (np.random.randn(num_actions), np.random.rand(num_actions))
        elif distribution == "heavy-tail":
            self.reward_parameters = np.random.rand(num_actions)
        else:
            print("Please use a supported reward distribution", flush = True)
            sys.exit(0)
        
        if distribution != "normal":
            self.optimal_arm = np.argmax(self.reward_parameters)
        else:
            self.optimal_arm = np.argmax(self.reward_parameters[0])
    
    def compute_gap(self, action):
        """
        Compute difference between reward of best possible action and the action taken
        """
        if self.distribution != "normal":
            gap = np.absolute(self.reward_parameters[self.optimal_arm] - self.reward_parameters[action])
        else:
            gap = np.absolute(self.reward_parameters[0][self.optimal_arm] - self.reward_parameters[0][action])
        return gap
    
    def step(self, action):
        """
        Compute one step of a Bandit environment with the action taken, producing a reward
        """
        
        valid_action = True
        if (action is None or action < 0 or action >= self.num_actions):
            print("Algorithm chose an invalid action; reset reward to -inf", flush = True)
            reward = float("-inf")
            valid_action = False
        
        if valid_action:
            if self.distribution == "bernoulli":
                reward = np.random.binomial(1, self.reward_parameters[action])
            elif self.distribution == "normal":
                reward = self.reward_parameters[0][action] + self.reward_parameters[1][action] * np.random.randn()
            elif self.distribution == "heavy-tail":
                reward = self.reward_parameters[action] + np.random.standard_cauchy()                
            else:
                print("Please use a supported reward distribution", flush = True)
                sys.exit(0)
            
        return(reward)
        


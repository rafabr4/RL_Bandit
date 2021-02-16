import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib import rcParams
from cycler import cycler
from scipy.stats import beta as dist_beta
import math
from matplotlib.ticker import StrMethodFormatter
import os
import sys

class Plot_Animation:
    """
    Parent class with some common parameters
    """
    
    def __init__(self, interval=5, fps=15):
        self.num_of_frames = None
        self.extra_frames = fps * 3 # Extend the animation at the end by 3 seconds
        self.interval = interval
        self.fps = fps
        self.blit = False
        self.animation_file_name = None
        self.fig = None
        # The path of the render needs to be adjusted accordingly
        rcParams['animation.ffmpeg_path'] = 'C:/ffmpeg-4.2-win64-static/bin/ffmpeg.exe'
    
class Arm_Counts_Animation(Plot_Animation):
    """
    Generates a plot with the times each action has been played, using
    a bar plot. The best action is showed with an arrow.
    """

    def __init__(self, ax, total_counts, best_action):
        Plot_Animation.__init__(self)
        self.total_counts = total_counts
        self.ax = ax
        self.best_action = best_action

    def update_plot(self, counts_vector):
        """
        This method is called for updating the frames for FuncAnimation
        It receives a vector with the counts of each action until now.
        """
        
        plt.sca(self.ax)
        
        k = len(counts_vector) # Number of actions
        x = range(k)
        bar_width = 3/5

        plt.cla()
        plt.bar(x, counts_vector, bar_width)
        
        plt.xlabel("Arm")
        plt.ylabel("Counts")
        plt.title("Arm selection")

        max_count = np.max(counts_vector)
        base = 100
        top_y_tick = math.ceil(max_count * 1.1 / base) * base # Round to multiples of base. Always have space between max_count and top

        # Add a marker to the best action
        marker_offset = 0.03 * top_y_tick
        plt.scatter(self.best_action, counts_vector[self.best_action] + marker_offset, marker="v", s=40, color="goldenrod", label="Optimal")

        plt.legend(loc=1,
                   fontsize="medium",
                   framealpha=0.8,
                   handletextpad=0.2,
                   handlelength=2)

        x_ticks = x
        y_ticks = self.make_y_ticks(top_y_tick, num_of_ticks=self.determine_num_of_y_ticks(base))
        plt.xticks(x_ticks)
        plt.yticks(y_ticks)

    def make_y_ticks(self, top_y_tick, num_of_ticks):
        # Make num_of_ticks ticks by dividing top_y_tick into equal ranges
        y_ticks = []
        for i in range(1, num_of_ticks + 1):
            factor = i / num_of_ticks
            new_tick = int(top_y_tick * factor)
            y_ticks.append(new_tick)
        
        return y_ticks
    
    def determine_num_of_y_ticks(self, base):
        if base % 4 == 0:
            return 4
        elif base % 5 == 0:
            return 5
        else:
            return 1

class Experiment_Regret_Animation(Plot_Animation):
    """
    Generates a plot with the accumulated regret over rounds.
    """
    
    def __init__(self, ax, regret_history):
        Plot_Animation.__init__(self)
        self.regret_history = regret_history
        self.ax = ax
        self.previous_top_x_tick = None
        self.previous_top_y_tick = None

    def update_plot(self, cumulative_regret):
        """
        This method is called for updating the frames for FuncAnimation
        It receives a vector with the cumulative regret until now.
        """
        
        plt.sca(self.ax)
        
        num_of_trials = len(cumulative_regret)
    
        plt.cla()
        plt.plot(cumulative_regret)
        plt.xlabel("Trial")
        plt.ylabel("Cumulative Regret")
        plt.title("Cumulative Regret over Trials")

        x_base = 100
        y_base = 5
        additional_space = 15
        # Round to multiples of x_base. Always have space between max_count and top
        top_x_tick = math.ceil((num_of_trials + additional_space) / x_base) * x_base
        
        # Update y ticks only when x tick changed
        if top_x_tick != self.previous_top_x_tick:
            # Make sure we're still within bounds
            if top_x_tick <= len(self.regret_history):
                max_value = self.regret_history[top_x_tick - 1]
                top_y_tick = math.ceil(max_value * 1.05 / y_base) * y_base  # Round to multiples of 10. Always have space between max_count and top
            elif self.previous_top_y_tick == None:
                # This happens when running for num_of_trials less than x_base
                max_value = self.regret_history[-1]
                top_y_tick = math.ceil(max_value * 1.05 / y_base) * y_base  # Round to multiples of 10. Always have space between max_count and top
            else:
                # This happens when top_x_tick is greater than the number of trials, for extra frames
                top_y_tick = self.previous_top_y_tick
        else:
            top_y_tick = self.previous_top_y_tick

        self.previous_top_x_tick = top_x_tick
        self.previous_top_y_tick = top_y_tick

        x_ticks = self.make_x_ticks(top_x_tick, num_of_ticks=self.determine_num_of_x_ticks(x_base))
        y_ticks = self.make_y_ticks(top_y_tick, num_of_ticks=5)
        plt.xticks(x_ticks)
        plt.yticks(y_ticks)
        self.ax.yaxis.set_major_formatter(StrMethodFormatter('{x: 5.1f}'))

    def make_x_ticks(self, top_x_tick, num_of_ticks):
        # Make num_of_ticks ticks by dividing top_x_tick into equal ranges
        x_ticks = []
        for i in range(1, num_of_ticks + 1):
            factor = i / num_of_ticks
            new_tick = int(top_x_tick * factor)
            x_ticks.append(new_tick)
        
        return x_ticks
    
    def make_y_ticks(self, top_y_tick, num_of_ticks):
        # Make num_of_ticks ticks by dividing top_y_tick into equal ranges
        y_ticks = []
        for i in range(1, num_of_ticks + 1):
            factor = i / num_of_ticks
            new_tick = top_y_tick * factor
            y_ticks.append(new_tick)
        
        return y_ticks
    
    def determine_num_of_x_ticks(self, base):
        if base % 4 == 0:
            return 4
        elif base % 5 == 0:
            return 5
        else:
            return 1

class Arm_True_Rewards_Animation(Plot_Animation):
    """
    Generates a plot with the true rewards and the observed average
    rewards of each action.
    In the case of using the UCB algorithm, it also plots the confidence interval.
    The best action is always differentiated with solid style.
    """
    
    def __init__(self, ax, algorithm, details, reward_parameters, average_rewards_history):
        Plot_Animation.__init__(self)
        self.algorithm_name = algorithm
        self.reward_parameters = reward_parameters
        self.average_rewards_history = average_rewards_history        
        self.ax = ax
        self.best_action = np.argmax(self.reward_parameters)

        if self.algorithm_name == "Optimistic Greedy":
            self.R = details["R"]
        
        self.previous_top_y_tick = np.inf

    def update_plot(self, rewards_data):
        """
        This method is called for updating the frames for FuncAnimation
        It receives a vector with the average rewards observed until now.
        In the case of UCB algorithm, it also receives the bonus vector
        which indicates the size of the confidence interval.
        """
        
        plt.sca(self.ax)

        average_rewards_vector = rewards_data["average_rewards_vector"]
        if self.algorithm_name == "UCB":
            bonus_vector = rewards_data["bonus_vector"]
        
        plt.cla()

        k = len(self.reward_parameters) # Number of actions
        x = np.array(range(k))

        max_true_reward = np.max(self.reward_parameters)
        indices_not_best_action = np.where(self.reward_parameters != max_true_reward)
        
        # Plot observed rewards (doing it in two steps to save only one label for all avg. reward markers)
        plt.scatter(x[0] - 0.02, average_rewards_vector[0], marker=".", s=60, color="#ff0000", label="Avg. Reward", zorder=4)
        plt.scatter(x - 0.02, average_rewards_vector, marker=".", s=60, color="#ff0000", zorder=4)
        
        # Plot true rewards (doing it in two steps to save only one label for all true reward markers)
        plt.scatter(x[0], self.reward_parameters[0], marker="_", s=80, color="#4C72B0", label="True Reward", zorder=3)
        plt.scatter(x, self.reward_parameters, marker="_", s=80, color="#4C72B0", zorder=3)

        # If this is UCB algorithm
        if self.algorithm_name == "UCB":
            # Plot confidence interval markers [top part]
            reward_plus_bonus = average_rewards_vector + bonus_vector
            plt.scatter(x, reward_plus_bonus, marker="_", s=30, color="#ff0000", zorder=2)
            
            # Plot confidence interval markers [bottom part]
            # Limit bonuses so the subtraction doesn't go below 0, for plotting purposes
            reward_minus_bonus = average_rewards_vector - bonus_vector
            reward_minus_bonus[reward_minus_bonus < 0.0] = -0.1
            plt.scatter(x, reward_minus_bonus, marker="_", s=30, color="#ff0000", zorder=2)

            # Plot confidence interval lines [top part] (only once we save the label)
            # Best action uses alpha = 1, not best actions use alpha different from 1
            self.ax.vlines(x=self.best_action,
                        ymin=reward_plus_bonus[self.best_action],
                        ymax=average_rewards_vector[self.best_action],
                        color="#ff0000",
                        linestyle="solid",
                        alpha=1,
                        zorder=2,
                        label="Confidence Interval")
            self.ax.vlines(x=x[indices_not_best_action],
                        ymin=reward_plus_bonus[indices_not_best_action],
                        ymax=average_rewards_vector[indices_not_best_action],
                        color="#ff0000",
                        linestyle="solid",
                        alpha=0.25,
                        zorder=2)

            # Plot confidence interval lines [bottom part]
            # Best action uses alpha = 1, not best actions use alpha different from 1
            self.ax.vlines(x=self.best_action,
                        ymin=reward_minus_bonus[self.best_action],
                        ymax=average_rewards_vector[self.best_action],
                        color="#ff0000",
                        linestyle="solid",
                        alpha=1,
                        zorder=2)
            self.ax.vlines(x=x[indices_not_best_action],
                        ymin=reward_minus_bonus[indices_not_best_action],
                        ymax=average_rewards_vector[indices_not_best_action],
                        color="#ff0000",
                        linestyle="solid",
                        alpha=0.25,
                        zorder=2)

        plt.title("Arm's Reward Distribution")
        plt.xlabel("Arm")
        if self.algorithm_name == "Optimistic Greedy":
            plt.ylabel("Reward")
        else:
            plt.ylabel("Probability")

        plt.xticks(x)

        
        if self.algorithm_name == "Optimistic Greedy":
            max_avg_reward = np.max(average_rewards_vector)
            yticks_values = self.make_y_ticks(max_avg_reward)
        elif self.algorithm_name == "UCB":
            max_reward_value = np.max(average_rewards_vector + bonus_vector)
            # In the first iteration(s), if we get a max of 0 then y limits are handled differently
            if max_reward_value == 0.0:
                yticks_values = self.make_y_ticks()
            else:
                yticks_values = self.make_y_ticks(max_reward_value)
        else:
            yticks_values = self.make_y_ticks()
        plt.yticks(yticks_values)
        
        # Set top ylim 4% higher than the top tick value
        self.ax.set_ylim(bottom=0, top=yticks_values[-1] * 1.04)

        # Plot lines all the way from 0 to max
        # Plot best action line with a different style
        plt.axvline(x=self.best_action,
                       color="#4C72B0",
                       linestyle="solid",
                       label="Optimal",
                       zorder=1)
        
        # Dummy line to set the sub-optimal label
        plt.axvline(color="#96b3e3",
                    linestyle=(0, (1, 3)),
                    label="Sub-Optimal",
                    zorder=1)
        
        # Plot all sub-optimal lines
        for i in indices_not_best_action[0]:
            plt.axvline(x=x[i],
                        color="#96b3e3",
                        linestyle=(0, (1, 3)),
                        zorder=1)
        
        if self.algorithm_name == "UCB":
            label_box_x_offset = -0.383
        else:
            label_box_x_offset = -0.017
        
        plt.legend(loc=2,
            scatterpoints=1,
            fontsize="medium",
            framealpha=1,
            handletextpad=0.2,
            handlelength=2,
            bbox_to_anchor=(label_box_x_offset, 1.24),
            ncol=5,
            columnspacing=1.23)

    def make_y_ticks(self, max_value=None):
        """
        Generates the y ticks values in different fashions according
        to the algorithm being used.
        """

        if (max_value is not None) and (max_value is not 0):
            if self.algorithm_name == "Optimistic Greedy":
                # Round to the next number that matches the order of magnitude
                # Eg. 0.8->1, 4.3->5, 64->70, 125->200, etc.
                top_y_tick = math.ceil(max_value)
                order = math.floor(np.log10(top_y_tick))
                power_of_10 = np.power(10, order)
                top_y_tick = math.ceil(top_y_tick/power_of_10) * power_of_10
                yticks = np.linspace(0.0, top_y_tick, num=6)
            elif self.algorithm_name == "UCB":
                # Round to multiples of base
                base = 0.5
                top_y_tick = math.ceil(max_value / base) * base
                # Don't allow the scale to go up after it had decreased, because it's confusing
                if top_y_tick > self.previous_top_y_tick:
                    top_y_tick = self.previous_top_y_tick
                # We only allow the scale to go up once, which is after the first k rounds where top_y_tick is going to be 1
                elif top_y_tick > 1:
                    self.previous_top_y_tick = top_y_tick
                yticks = np.linspace(0.0, top_y_tick, num=6)
        else:
            # Greedy and Epsilon Greedy never go higher than 1
            yticks = np.linspace(0.0, 1.0, num=6)

        return yticks

class Arm_Beta_Animation(Plot_Animation):
    """
    Generates a plot with the Beta distributions of the best n actions,
    including its true reward as a dashed line. It is intended to be used
    with the Thompson Beta algorithm.
    """
    
    def __init__(self, ax, reward_parameters, best_n_actions):
        Plot_Animation.__init__(self)
        self.reward_parameters = reward_parameters
        self.best_n_actions = best_n_actions
        self.ax = ax
        self.best_action = np.argmax(self.reward_parameters)

        self.previous_top_y_tick = 0

    def update_plot(self, rewards_data):
        """
        This method is called for updating the frames for FuncAnimation
        It receives vectors with successes and failures, used for
        updating the Beta distribution.
        """
        
        plt.sca(self.ax)

        successes_vector = rewards_data["successes_vector"]
        failures_vector = rewards_data["failures_vector"]
        
        plt.cla()

        k = len(self.best_n_actions)  # Number of actions to plot
        
        x_range = 1000
        x = np.linspace(0, 1.0, x_range)

        # Using default colors, that were changed because we're using a seaborn style
        colors = cycler(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                               '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
        rcParams['axes.prop_cycle'] = colors

        # Loop through best n actions
        dist = {}
        y = np.zeros((k, x_range))
        for i in range(k):
            # Create the Beta distribution
            dist[i] = dist_beta(successes_vector[i], failures_vector[i])
            # Plot the PDF of the distribution
            y[i,:] = dist[i].pdf(x)
            current_curve, = plt.plot(x,
                                     y[i,:],
                                     label="Arm " + str(self.best_n_actions[i]),
                                     zorder=k-i) # Best plot goes on top
            current_color = current_curve.get_color()

            # Shade area under the curve
            plt.fill_between(x, y[i,:], alpha=0.4, color=current_color)

            # Plot a line of the true mean reward
            self.ax.vlines(x=self.reward_parameters[self.best_n_actions[i]],
                          ymin=0,
                          ymax=dist[i].pdf(self.reward_parameters[self.best_n_actions[i]]),
                          color=current_color,
                          linestyle=(0, (2, 3)))

        plt.xlim((0,1.0))

        # Set the y top limit to 5% more than the max value to be plotted
        max_y = math.ceil(np.max(y) * 1.05)
        # Don't allow the y axis to shrink
        if max_y < self.previous_top_y_tick:
            max_y = self.previous_top_y_tick
        else:
            self.previous_top_y_tick = max_y
        y_ticks = np.linspace(0, max_y, 5)

        plt.yticks(y_ticks)

        plt.xlabel("Mean Reward")
        plt.ylabel("PDF")
        plt.title("Mean Rewards Beta Distributions")

        self.ax.yaxis.set_major_formatter(StrMethodFormatter('{x: 5.1f}'))
        
        # For some reason we need to specify 0 as the lower y limit
        self.ax.set_ylim(bottom=0)

        handles, labels = self.ax.get_legend_handles_labels()
        # Sort labels alphabetically, since we had plotted them in zorder of best to worst
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))

        # Show the legends
        plt.legend(loc=2,
            scatterpoints=1,
            fontsize="medium",
            framealpha=1,
            handletextpad=0.2,
            handlelength=2,
            bbox_to_anchor=(0.017, 1.24),
            ncol=5,
            columnspacing=1.23)

class Plot_Details_Animation(Plot_Animation):
    """
    Generates a stationary plot with the name and the details of the algorithm
    """
    
    def __init__(self, ax, algorithm, details):
        Plot_Animation.__init__(self)
        self.ax = ax
        self.algorithm_name = algorithm

        if self.algorithm_name == "Round Robin":
            self.max_round_robin_rounds = details["nrr_rounds"]
        elif self.algorithm_name == "Epsilon Greedy":
            self.epsilon = details["epsilon"]
        elif self.algorithm_name == "Optimistic Greedy":
            self.R = details["R"]
        elif self.algorithm_name == "UCB":
            self.C = details["C"]

        self.setup_plot()
    
    def setup_plot(self):
        plt.sca(self.ax)
        
        # build a rectangle in axes coords
        left, width = 0.25, 0.5
        bottom, height = 0.25, 0.5
        right = left + width
        top = bottom + height

        if self.algorithm_name == "Round Robin" or \
           self.algorithm_name == "Epsilon Greedy" or \
           self.algorithm_name == "Optimistic Greedy" or \
           self.algorithm_name == "UCB":
            h_offset = 0.07
            plot_details = True
        else:
            # Greedy and Thompson Beta have no details to show
            h_offset = 0
            plot_details = False

        self.ax.text(0.5 * (left + right), 0.5 * (bottom + top) + h_offset,
                     self.algorithm_name,
                     horizontalalignment='center',
                     verticalalignment='center',
                     fontsize=20, color='black',
                     fontweight="bold",
                     transform=self.ax.transAxes)
        
        if self.algorithm_name == "Round Robin":
            details_string = r"$\bullet n_{RR} = " + str(self.max_round_robin_rounds) + r"$"
        elif self.algorithm_name == "Epsilon Greedy":
            details_string = r"$\bullet \epsilon = " + str(self.epsilon) + r"$"
        elif self.algorithm_name == "Optimistic Greedy":
            details_string = r"$\bullet R = " + str(self.R) + r"$"
        elif self.algorithm_name == "UCB":
            details_string = r"$\bullet C = " + str(self.C) + r"$"

        if plot_details:
            self.ax.text(0.5 * (left + right), 0.5 * (bottom + top) - h_offset,
                        details_string,
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=20, color='black',
                        transform=self.ax.transAxes)

class All_Plots_Animation(Plot_Animation):
    """
    Generates 4 animated subplots:
        - Details: shows name and parameters of the algorithm used
        - True Rewards: shows the true rewards of the actions
        - Regret: shows the accumulated regret over rounds
        - Counts: shows how many times each action has been picked
    """
    
    def __init__(self, data, blit=False, interval=5, fps=10):
        """
        Initialize all necessary parameters and objects for animation
            - algorithm: name of the algorithm
            - details: algorithm specific parameters
            - total_counts: history of how actions were played (num of trials, num of actions)
            - regret_history: history of regret accumulation (num of trials,)
            - reward_parameters: true rewards for each action (num of actions,)
            - average_Rewards_history: history of observed average rewards (num of trials, num of actions)
            * bonus_history: history of confidence intervals for UCB (num of trials, num of actions)
            * successes_history: history of action successes for Thompson Beta (num of trials, num of actions)
            * failures_history: history of action failures for Thompson Beta (num of trials, num of actions)
        """

        Plot_Animation.__init__(self, interval, fps)

        self.algorithm = data["algorithm"]
        self.details = data["details"]
        self.total_counts = data["total_counts"]
        self.regret_history = data["regret_history"]
        self.reward_parameters = data["reward_parameters"]
        self.average_rewards_history = data["average_rewards_history"]
        if self.algorithm == "UCB":
            self.bonus_history = data["bonus_history"]
        elif self.algorithm == "Thompson Beta":
            self.successes_history = data["successes_history"]
            self.failures_history = data["failures_history"]
            # Only best n actions will be animated in the Beta distribution plot
            n = 5
            self.best_n_actions = self.keep_top_n_actions(n)
        self.best_action = np.argmax(self.reward_parameters)

        self.num_of_trials = self.total_counts.shape[0]
        # Animation has one frame by trial plus an extension at the end for easier inspection
        self.num_of_frames = self.num_of_trials + self.extra_frames
        self.animation_file_name = self.algorithm + '_' + str(fps) + 'fps.mp4'

        plt.style.use("seaborn-deep")

        self.fig = plt.figure(figsize=(12.5, 8))

        # Define 4 subplots
        self.ax_details = plt.subplot(221)
        self.ax_true_rewards = plt.subplot(222)
        self.ax_regret = plt.subplot(223)
        self.ax_arm_counts = plt.subplot(224)
        
        # Set up plot for the details of the algorithm
        self.ax_details.axis('off')
        self.details_animation = Plot_Details_Animation(self.ax_details,
                                                        self.algorithm,
                                                        self.details)
        
        # Set up the plot for the true rewards of the actions
        if self.algorithm == "Thompson Beta":
            self.true_rewards_animation = Arm_Beta_Animation(self.ax_true_rewards,
                                                             self.reward_parameters,
                                                             self.best_n_actions)
        else:
            self.true_rewards_animation = Arm_True_Rewards_Animation(self.ax_true_rewards,
                                                                     self.algorithm,
                                                                     self.details,
                                                                     self.reward_parameters,
                                                                     self.average_rewards_history)
        
        # Set up the plot for the counts of the actions
        self.arm_counts_animation = Arm_Counts_Animation(self.ax_arm_counts,
                                                         self.total_counts,
                                                         self.best_action)

        # Set up the plot for visualizing the regret
        self.regret_animation = Experiment_Regret_Animation(self.ax_regret,
                                                            self.regret_history)

    def generate_animation(self, render_action="show"):
        """
        Generates the animation that can be plotted or saved to a file.
        """
        # update_plot receives parameters that are output by frames_func
        ani = animation.FuncAnimation(self.fig,
                        self.update_plot,
                        frames=self.frames_func,
                        blit=self.blit,
                        repeat=False,
                        interval=self.interval,
                        save_count=self.num_of_frames)
        plt.tight_layout(pad=5, w_pad=1.8, h_pad=3.0)

        if render_action == "save":
            self.save_animation(ani)
        elif render_action == "show":
            self.show_animation()
    
    def save_animation(self, ani):
        ani.save(self.animation_file_name, writer=animation.FFMpegWriter(fps=self.fps))
        os.startfile(self.animation_file_name) # Automatically opens video file after finishing rendering
        
    def show_animation(self):
        plt.show()
    
    def update_plot(self, update_data):
        """
        Calls the update_plot methods of each subplot, passing
        the appropriate data needed for updating the plots.
        This is called for each frame of FuncAnimation.
        """

        counts_vector = update_data["counts_vector"]
        self.arm_counts_animation.update_plot(counts_vector)

        cumulative_regret = update_data["cumulative_regret"]
        self.regret_animation.update_plot(cumulative_regret)

        rewards_data = {}
        rewards_data["average_rewards_vector"] = update_data["average_rewards_vector"]
        if self.algorithm == "UCB":
            rewards_data["bonus_vector"] = update_data["bonus_vector"]
        elif self.algorithm == "Thompson Beta":
            rewards_data["successes_vector"] = update_data["successes_vector"]
            rewards_data["failures_vector"] = update_data["failures_vector"]
        self.true_rewards_animation.update_plot(rewards_data)

    def frames_func(self):
        """
        Prepares the information to be sent to the subplots in each frame,
        by retrieving the data up to the current trial.
        At the end, the same values are sent for some extra frames where
        the subplots will not be moving.
        This is called by FuncAnimation at each frame.
        """

        update_data = {}
        
        # Go through all trials
        for i in range(self.num_of_trials):
            update_data["counts_vector"] = self.total_counts[i,:]
            update_data["cumulative_regret"] = self.regret_history[:i + 1]
            update_data["average_rewards_vector"] = self.average_rewards_history[i,:]
            if self.algorithm == "UCB":
                update_data["bonus_vector"] = self.bonus_history[i,:]
            if self.algorithm == "Thompson Beta":
                update_data["successes_vector"] = self.successes_history[i,:]
                update_data["failures_vector"] = self.failures_history[i,:]
            yield update_data
        
        # Yield the last values for some extra frames
        for i in range(self.extra_frames):
            update_data["counts_vector"] = self.total_counts[-1,:]
            update_data["cumulative_regret"] = self.regret_history
            update_data["average_rewards_vector"] = self.average_rewards_history[-1,:]
            if self.algorithm == "UCB":
                update_data["bonus_vector"] = self.bonus_history[-1,:]
            if self.algorithm == "Thompson Beta":
                update_data["successes_vector"] = self.successes_history[-1,:]
                update_data["failures_vector"] = self.failures_history[-1,:]
            yield update_data
    
    def keep_top_n_actions(self, n):
        """
        This function searches for the best n actions in the success/failure history and keeps them
        Only intended to use with Thompson Beta algorithm
        """

        best_n_actions = []
        # Grab the vector corresponding to the last trial (index -1)
        final_successes_vector = np.copy(self.successes_history[-1,:])
        for _ in range(n):
            current_best_action = np.argmax(final_successes_vector)
            best_n_actions.append(current_best_action)
            # Set this action's successes to 0 so the argmax doesn't find it again
            final_successes_vector[current_best_action] = 0
        self.successes_history = self.successes_history[:, best_n_actions]
        self.failures_history = self.failures_history[:, best_n_actions]
        return best_n_actions

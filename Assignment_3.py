# https://towardsdatascience.com/policy-gradient-methods-104c783251e0
import multiprocessing
import os

import gym
import numpy as np
import torch
import time
from Helper import LearningCurvePlot, smooth


class REINFORCE_Agent:
    """ Agent that uses Monte Carlo Policy-Gradient Methods. """
    def __init__(self, env, param_dict: dict):
        """ Set parameters and initialize neural network and optimizer. """
        self.n_states = env.observation_space.shape[0]
        self.n_actions = env.action_space.n
        self.learning_rate = param_dict['alpha']
        self.gamma = param_dict['gamma']

        self.memory = []  # used for memorizing traces

        self.model = self._initialize_nn()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def _initialize_nn(self):
        """ Initialize neural network. """
        print('Create a neural network with {} input nodes and {} output nodes'.format(self.n_states, self.n_actions))

        ## TO DO: Have a look at optimizing these model specifications ##
        model = torch.nn.Sequential(
            torch.nn.Linear(self.n_states, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, self.n_actions),
            torch.nn.Softmax(dim=0))

        return model  # return initialized model

    def memorize(self, s, a, r):
        """ Memorize state, action and reward. """
        self.memory.append((s, a, r))

    def forget(self):
        """ Empty memory. """
        self.memory = []

    def choose_action(self, s):
        """ Return action and probability distribution given state. """
        action_probability = self.model(torch.from_numpy(s).float()).detach().numpy()
        action = np.random.choice(np.arange(0, self.n_actions), p=action_probability)
        return action

    def loss (self):
        """ Return the loss """
        # Estimate the return of the trace
        R_t = torch.Tensor([r for (s, a, r) in self.memory]).flip(dims=(0,))  ## Is this the R_t? print it ##

        R_list = []
        # Total return per timestep of the trace
        for i in range(len(self.memory)):
            R_elements = [pow(self.gamma, idx - i - 1) * R_t[idx - 1].numpy() for idx in range(i + 1, len(self.memory) + 1)]
            R_list.append(sum(R_elements))

        U_theta = torch.FloatTensor(R_list)
        U_theta = U_theta / U_theta.max()

        all_states = torch.Tensor(np.array([s for (s, a, r) in self.memory]))
        all_actions = torch.Tensor(np.array([a for (s, a, r) in self.memory]))

        predictions = self.model(all_states)
        probabilities = predictions.gather(dim=1, index=all_actions.long().view(-1, 1)).squeeze()

        # Update the weights of the policy
        loss = - torch.sum(torch.log(probabilities) * U_theta)
        return loss



def act_in_env(n_traces: int, n_timesteps: int, param_dict: dict):
    env = gym.make('CartPole-v1')  # create environment of CartPole-v1
    agent = REINFORCE_Agent(env, param_dict)  # initiate the agent

    env_scores = []  # shows trace length over training time
    for m in range(n_traces):
        state = env.reset()  # reset environment and get initial state

        # Use the policy to collect a trace
        for t in range(n_timesteps):
            action = agent.choose_action(s=state)
            state_next, _, done, _ = env.step(action)
            agent.memorize(s=state, a=action, r=t + 1)
            state = state_next

            if done:
                env_scores.append(t + 1)
                break

        loss = agent.loss()

        agent.optimizer.zero_grad()
        loss.backward()
        agent.optimizer.step()

        if m % 50 == 0 and m > 0:
            print('Trace {}\tAverage Score: {:.2f}'.format(m, np.mean(env_scores[-50:-1])))

        agent.forget()
    env.close()
    return env_scores



def average_over_repetitions(n_repetitions, n_traces, n_timesteps, param_dict, smoothing_window):

    reward_results = np.empty([n_repetitions,n_traces]) # Result array
    now = time.time()

    for rep in range(n_repetitions):
        print(f'Repetition: {rep}')
        rewards = act_in_env(n_traces=n_traces, n_timesteps=n_timesteps, param_dict=param_dict)
        reward_results[rep] = rewards

    print('Running one setting takes {} minutes'.format((time.time() - now) / 60))
    learning_curve = np.mean(reward_results, axis=0)  # average over repetitions
    standard_error = np.std(reward_results, axis=0)
    learning_curve = smooth(learning_curve, smoothing_window)  # additional smoothing
    standard_error = smooth(standard_error, smoothing_window)
    return learning_curve, standard_error


def experiment( ):

    n_repetitions = 20
    n_traces = 1000
    n_timesteps = 500

    # param_dict = {
    #     'alpha': 0.001,  # Learning-rate
    #     'gamma': 0.99  # Discount factor
    # }

    smoothing_window = 101

    ### Method: REINFORCE
    method = 'REINFORCE'
    gamma = 0.99
    learning_rates = [0.001, 0.025, 0.01]
    colours = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]

    run = 0
    Plot = LearningCurvePlot(title=f'Method: {method} --- Results averaged of {n_repetitions} repetitions')
    for alpha in learning_rates:

            param_dict = {
                'alpha': alpha,  # Learning-rate
                'gamma': gamma  # Discount factor
            }

            print(f'Running {method}-method with learning rate={alpha} and discount rate={gamma}')


            learning_curve, standard_error = average_over_repetitions(n_repetitions, n_traces, n_timesteps, param_dict, smoothing_window)
            Plot.add_curve(x=np.arange(1, len(learning_curve)+1),
                           y=learning_curve,
                           std=standard_error,
                           col=colours[run],
                           label=r'REINFORCE with $\alpha$={} and $\gamma$={}'.format(alpha, gamma))
            run += 1
    Plot.save('REINFORCE.png')

if __name__ == '__main__':
    experiment()





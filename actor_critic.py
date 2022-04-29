# https://towardsdatascience.com/policy-gradient-methods-104c783251e0
import gym
import numpy as np
import torch

class Actor_Critic_Agent:
    """ Actor-critic policy gradient used. """
    def __init__(self, env, param_dict: dict):
        """ Set parameters and initialize neural network and optimizer. """
        self.n_states = env.observation_space.shape[0]
        self.n_actions = env.action_space.n
        
        self.learning_rate_actor = param_dict['alpha_1']
        self.learning_rate_critic = param_dict['alpha_2']
        self.n_depth = param_dict['n_depth']

        self.memory = []    # used for memorizing traces
        self.Q_values = []  # used for memorizing the Q-values
        
        self.model_actor = self._initialize_nn(type='actor')
        self.model_critic = self._initialize_nn(type='critic')
        
        self.optimizer_actor = torch.optim.Adam(self.model_actor.parameters(), lr = self.learning_rate_actor)
        self.optimizer_critic = torch.optim.Adam(self.model_critic.parameters(), lr = self.learning_rate_critic)

    def _initialize_nn(self, type: str = 'actor'):
        """ Initialize neural network. """
        print('Create a neural network with {} input nodes and {} output nodes'.format(self.n_states, self.n_actions))

        ## TO DO: Have a look at optimizing these model specifications ##
        if type == 'actor':
            model = torch.nn.Sequential(
                torch.nn.Linear(self.n_states, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, self.n_actions),
                torch.nn.Softmax(dim=0))

        elif type == 'critic':
            model = torch.nn.Sequential(
                torch.nn.Linear(self.n_states, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, self.n_actions))
        return model    # return initialized model
    
    def memorize(self, s, a, r):
        """ Memorize state, action and reward. """
        self.memory.append((s, a, r))
        
    def forget(self):
        """ Empty memory. """
        self.memory = []

    def forget_Q_values (self):
        """ Empty memory of Q_values """
        self.Q_values = []

    def choose_action(self, s):
        """ Return action and probability distribution given state. """
        action_probability = self.model_actor(torch.from_numpy(s).float()).detach().numpy()
        action = np.random.choice(np.arange(0, self.n_actions), p=action_probability)
        return action

    def Q_values_calc (self, t):
        """ Save the Q-values for every timestep until n_depth in memory"""
        r_t = torch.Tensor([r for (s, a, r) in self.memory]).flip(dims=(0,))
        s_t = torch.Tensor(np.array([s for (s, a, r) in self.memory]))

        Q_value = 0
        # Total return per timestep of the trace
        for k in range(self.n_depth):
            Q_value += r_t[t + k] + self.model_critic(s_t[t + self.n_depth])
        self.Q_values.append(Q_value)

    def loss (self):
        """ Return the loss """
        a_t = torch.Tensor(np.array([a for (s, a, r) in self.memory]))
        s_t = torch.Tensor(np.array([s for (s, a, r) in self.memory]))

        predictions = self.model_actor(s_t)
        probabilities = predictions.gather(dim=1, index=a_t.long().view(-1, 1)).squeeze()

        # Update the weights of the policy
        loss_actor = - self.learning_rate_actor * torch.sum(self.Q_values * torch.sum(torch.log(probabilities)))
        loss_critic = self.learning_rate_critic * torch.sum(self.Q_values - self.model_critic(s_t))**2
        self.forget_Q_values()

        return loss_actor, loss_critic


def act_in_env(epochs: int, n_traces: int, n_timesteps: int, param_dict: dict):
    env = gym.make('CartPole-v1')                   # create environment of CartPole-v1
    agent = Actor_Critic_Agent(env, param_dict)     # initiate the agent

    for e in range(epochs):
        env_scores = []                     # shows trace length over training time
        for m in range(n_traces):
            state = env.reset()             # reset environment and get initial state

            ##### Use current policy to collect a trace
            for t in range(n_timesteps):
                action = agent.choose_action(s=state)  
                state_next, _, done, _ = env.step(action)
                agent.memorize(s=state, a=action, r=t+1)
                state = state_next

                if done:
                    for t in range(n_timesteps):
                        agent.Q_values_calc(t=t)
                    env_scores.append(t+1)
                    break

            loss_actor, loss_critic = agent.loss()

            # Update actor NN
            agent.optimizer_actor.zero_grad()
            loss_actor.backward()
            agent.optimizer_actor.step()

            # Update critic NN
            agent.optimizer_critic.zero_grad()
            loss_critic.backward()
            agent.optimizer_critic.step()

        print('Epoch {}: {}'.format(e, env_scores))
        agent.forget()
        env.close()
        return env_scores

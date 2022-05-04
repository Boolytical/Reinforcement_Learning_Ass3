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
        self.option = param_dict['option']

        self.memory = []    # used for memorizing traces
        self.psi_values = []  # used for memorizing the psi-values
        
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

    def forget_psi_values(self):
        """ Empty memory of Q_values """
        self.psi_values = []

    def choose_action(self, s):
        """ Return action and probability distribution given state. """
        action_probability = self.model_actor(torch.from_numpy(s).float()).detach().numpy()
        if np.isnan(action_probability).any():
            print(np.isnan(action_probability).any())
            print(torch.min(self.model_actor(torch.from_numpy(s).float())))
            print(torch.max(self.model_actor(torch.from_numpy(s).float())))
            print(f'States: {s} --- Action Probability: {action_probability}')
        action = np.random.choice(np.arange(0, self.n_actions), p=action_probability)
        return action

    def calculate_psi(self, t):
        """ Calculate and save the potential choices psi """
        r_t = torch.Tensor([r for (s, a, r) in self.memory]).flip(dims=(0,))
        s_t = torch.Tensor(np.array([s for (s, a, r) in self.memory]))

        # TODO: avoid that index exceeds length of memorized states
        if self.option == 'bootstrapping':
            if t + self.n_depth < len(self.memory):
                s_t_depth = torch.Tensor(s_t[t + self.n_depth])
                expected_return_per_action = self.model_critic(s_t_depth)  # outputs return values for action 0 and 1
                value = expected_return_per_action.max()  # to get value of s_t+n, get highest return of output nodes

                Q_value = 0  # Q_n(s_t, a_t)
                # Total return per timestep of the trace
                for k in range(self.n_depth - 1):
                    Q_value += r_t[t + k] + value
                self.psi_values.append(Q_value)

            else:
                self.Q_values.append(0)  # TODO: what to do with these cases?

        elif self.option == 'baseline_subtraction':
            expected_return_per_action_t = self.model_critic(s_t[t])  # Estimated value at timestep t
            value_substract = expected_return_per_action_t.max()

            A_val = 0
            for k in range(t, len(self.memory)):  # Compute Q-value
                A_val += r_t[k] - value_substract

            self.psi_values.append(A_val)

        elif self.option == 'bootstrapping_baseline':
            if t + self.n_depth < len(self.memory):
                s_t_depth = torch.Tensor(s_t[t + self.n_depth])
                expected_return_per_action = self.model_critic(s_t_depth)  # outputs return values for action 0 and 1
                value = expected_return_per_action.max()

                expected_return_per_action_t = self.model_critic(s_t[t])  # Estimated value at timestep t
                value_substract = expected_return_per_action_t.max()

                A_val = 0
                for k in range(self.n_depth - 1):
                    A_val += r_t[t + k] + value - value_substract
                self.psi_values.append(A_val)

            else:
                self.psi_values.append(0)

    def loss(self):
        """ Return the loss for actor and critic. """
        a_t = torch.Tensor(np.array([a for (s, a, r) in self.memory]))
        s_t = torch.Tensor(np.array([s for (s, a, r) in self.memory]))

        predictions = self.model_actor(s_t)
        probabilities = predictions.gather(dim=1, index=a_t.long()
                                           .view(-1, 1)).squeeze()  # get policy pi_actor(a_t|s_t) for each timestep

        ##### Update the weights of the policy
        all_return_values = self.model_critic(s_t)
        value_t = torch.mean(all_return_values, 1)
        psi = torch.Tensor(self.psi_values)

        loss_actor = - torch.sum(psi * torch.sum(torch.log(probabilities)))
        loss_critic = torch.sum(pow(psi - value_t, 2))

        self.forget_psi_values()
        return loss_actor, loss_critic


def act_in_env(epochs: int, n_traces: int, n_timesteps: int, param_dict: dict):
    env = gym.make('CartPole-v1')                   # create environment of CartPole-v1
    agent = Actor_Critic_Agent(env, param_dict)     # initiate the agent
    
    avg_per_epoch = []
    for e in range(epochs):
        env_scores = []                     # shows trace length over training time TODO: probably need to collect per epoch as well, otherwise only last epoch rewards are returned
        for m in range(n_traces):
            state = env.reset()             # reset environment and get initial state

            ##### Use current policy to collect a trace
            for t in range(n_timesteps):
                action = agent.choose_action(s=state)  
                state_next, reward, done, _ = env.step(action)
                agent.memorize(s=state, a=action, r=t+reward)
                state = state_next

                if done: # Calculate the potential choices
                    for t in range(len(agent.memory)):
                        agent.calculate_psi(t=t)
                        
                    env_scores.append(t+1)
                    break

            loss_actor, loss_critic = agent.loss()  # after trace is done, calculate loss for actor and critic

            ##### After every trace, update networks
            # Update actor NN
            agent.optimizer_actor.zero_grad()
            loss_actor.backward()
            agent.optimizer_actor.step()

            # Update critic NN
            agent.optimizer_critic.zero_grad()
            loss_critic.backward()
            agent.optimizer_critic.step()
            
            agent.forget()
        
        if e % 50 == 0 and e > 0:
            print('Epoch {}     Average Score: {}'.format(e, np.mean(env_scores)))
        avg_per_epoch.append(np.mean(env_scores))
        
    env.close()
    return avg_per_epoch
    
    
##### Quick way to test #####
"""
param_dict = {
    'alpha_1': 0.001,
    'alpha_2': 0.001,
    'n_depth': 4,
    'option': 'bootstrapping'}

act_in_env(epochs=1, n_traces=5, n_timesteps=500, param_dict=param_dict)
"""

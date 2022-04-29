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

        self.memory = []    # used for memorizing traces
        
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

    def choose_action(self, s):
        """ Return action and probability distribution given state. """
        action_probability = self.model_actor(torch.from_numpy(s).float()).detach().numpy()
        action = np.random.choice(np.arange(0, self.n_actions), p=action_probability)
        return action

    def loss (self):
        """ Return the loss """
        # Estimate the return of the trace
        R_t = torch.Tensor([r for (s, a, r) in self.memory]).flip(dims=(0,))

        R_list = []
        # Total return per timestep of the trace
        for i in range(len(self.memory)):
            R_elements = [pow(self.gamma, idx - i - 1) * R_t[idx - 1].numpy() for idx in range(i + 1, len(self.memory) + 1)]
            R_list.append(sum(R_elements))

        U_theta = torch.FloatTensor(R_list) # Expected reward
        U_theta = U_theta / U_theta.max() # Normalized expected reward

        all_states = torch.Tensor(np.array([s for (s, a, r) in self.memory]))
        all_actions = torch.Tensor(np.array([a for (s, a, r) in self.memory]))

        predictions = self.model(all_states)
        probabilities = predictions.gather(dim=1, index=all_actions.long().view(-1, 1)).squeeze()

        # Update the weights of the policy
        loss = - torch.sum(torch.log(probabilities) * U_theta)
        return loss



def act_in_env(epochs: int, n_traces: int, n_timesteps: int, param_dict: dict):
    env = gym.make('CartPole-v1')                   # create environment of CartPole-v1
    agent = Actor_Critic_Agent(env, param_dict)     # initiate the agent

    for e in range(epochs):
        scores_per_trace = []                     # shows trace length over training time
        
        gradient =  0   # initial gradient
        for m in range(n_traces):
            state = env.reset()             # reset environment and get initial state

            ##### Use current policy to collect a trace
            for t in range(n_timesteps):
                action = agent.choose_action(s=state)  
                state_next, _, done, _ = env.step(action)
                agent.memorize(s=state, a=action, r=t+1)
                state = state_next

                if done:
                    scores_per_trace.append(t+1)
                    break
            
            r_t = torch.Tensor([r for (s, a, r) in agent.memory]).flip(dims=(0, ))
            a_t = torch.Tensor(np.array([a for (s, a, r) in agent.memory]))
            s_t = torch.Tensor(np.array([s for (s, a, r) in agent.memory]))
            
            R = 0
            for t in range(len(agent.memory)):
                print(t)
                # gradient_actor -= R * torch.log(probability)  # minimizing 
                # gradient_critic
            
        #agent.optimizer.zero_grad()
        #gradient.backward()
        #agent.optimizer.step()
        
        print('Epoch {}: {}'.format(e, scores_per_trace))
        
    env.close()
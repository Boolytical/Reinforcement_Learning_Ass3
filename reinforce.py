# https://towardsdatascience.com/policy-gradient-methods-104c783251e0
import gym
import numpy as np
import torch

class REINFORCE_Agent:
    """ Agent that uses Monte Carlo Policy-Gradient Methods. """
    def __init__(self, env, param_dict: dict):
        """ Set parameters and initialize neural network and optimizer. """
        self.n_states = env.observation_space.shape[0]
        self.n_actions = env.action_space.n
        self.learning_rate = param_dict['alpha']
        self.gamma = param_dict['gamma']
        
        self.memory = []    # used for memorizing traces
        
        self.model = self._initialize_nn()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.learning_rate)

    def _initialize_nn(self):
        """ Initialize neural network. """
        print('Create a neural network with {} input nodes and {} output nodes'.format(self.n_states, self.n_actions))

        ## TO DO: Have a look at optimizing these model specifications ##
        model = torch.nn.Sequential(
            torch.nn.Linear(self.n_states, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, self.n_actions),
            torch.nn.Softmax(dim=0))

        return model    # return initialized model
    
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



def act_in_env(epochs: int, n_traces: int, n_timesteps: int, param_dict: dict):
    env = gym.make('CartPole-v1')               # create environment of CartPole-v1
    agent = REINFORCE_Agent(env, param_dict)    # initiate the agent

    
    
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
            for t in reversed(range(len(agent.memory))):
                R = r_t[t] + agent.gamma * R    # go backwards through whole trace
                
                prediction = agent.model(s_t[t])
                probability = prediction[a_t[t].long()]
                
                gradient -= R * torch.log(probability)  # minimizing 
            
        agent.optimizer.zero_grad()
        gradient.backward()
        agent.optimizer.step()
        
        print('Epoch {}: {}'.format(e, scores_per_trace))
        
    env.close()

param_dict = {
    'alpha': 0.001,             # Learning-rate
    'gamma': 0.99               # Discount factor
}

act_in_env(epochs=1000, n_traces=1, n_timesteps=500, param_dict=param_dict)

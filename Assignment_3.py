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
        return action_probability, action



def act_in_env(n_traces: int, n_timesteps: int, param_dict: dict):
    env = gym.make('CartPole-v1')               # create environment of CartPole-v1
    agent = REINFORCE_Agent(env, param_dict)    # initiate the agent

    env_scores = []                     # shows trace length over training time
    for m in range(n_traces):
        state = env.reset()             # reset environment and get initial state

        # Use the policy to collect a trace
        for t in range(n_timesteps):
            action_probability, action = agent.choose_action(s=state)   ## TODO action_probability not used yet, maybe consider using this instead of recalculating in line 87 ##
            state_next, _, done, _ = env.step(action)
            agent.memorize(s=state, a=action, r=t+1)
            
            state = state_next

            if done:
                env_scores.append(t+1)
                break

        # Estimate the return of the trace
        R_t = torch.Tensor([r for (s, a, r) in agent.memory]).flip(dims=(0, )) ## Is this the R_t? print it ##
        G_k = []                                            # Total return per timestep of the trace
        for i in range(len(agent.memory)):
            new_G_k = 0
            power = 0
            for j in range(i, len(agent.memory)):
                new_G_k = new_G_k + ((agent.gamma ** power) * R_t[j]).numpy()
                power += 1
            G_k.append(new_G_k)
            
        U_theta = torch.FloatTensor(G_k)                    # Expected return
        U_theta /= U_theta.max()                            ## Why is this needed? ##

        all_states = torch.Tensor(np.array([s for (s, a, r) in agent.memory]))
        all_actions = torch.Tensor(np.array([a for (s, a, r) in agent.memory]))

        predictions = agent.model(all_states)
        probabilities = predictions.gather(dim=1, index=all_actions.long().view(-1, 1)).squeeze()

        # Update the weights of the policy
        loss = - torch.sum(torch.log(probabilities) * U_theta)

        agent.optimizer.zero_grad()
        loss.backward()
        agent.optimizer.step()

        if m % 50 == 0 and m > 0:
            print('Trace {}\tAverage Score: {:.2f}'.format(m, np.mean(env_scores[-50:-1])))
            
        agent.forget()
        
    env.close()
    return env_scores



param_dict = {
    'alpha': 0.001,             # Learning-rate
    'gamma': 0.99               # Discount factor
}

act_in_env(n_traces=1000, n_timesteps=500, param_dict=param_dict)

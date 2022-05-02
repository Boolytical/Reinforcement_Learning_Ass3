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
        return model

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

    def loss(self):
        """ Calculate the loss for the trace """
        # Estimate the return of the trace
        R_t = torch.Tensor([r for (s, a, r) in self.memory]).flip(dims=(0, ))

        G_k = []                                            # Total return per timestep of the trace
        for i in range(len(self.memory)):
            new_G_k = 0
            power = 0
            for j in range(i, len(self.memory)):
                new_G_k = new_G_k + ((self.gamma ** power) * R_t[j]).numpy()
                power += 1
            G_k.append(new_G_k)

        U_theta = torch.FloatTensor(G_k)                    # Expected return
        U_theta = U_theta / U_theta.max()                   # Normalized expected return; more stable

        ### Determine? What are we doing here? ###
        all_states = torch.Tensor(np.array([s for (s, a, r) in self.memory]))
        all_actions = torch.Tensor(np.array([a for (s, a, r) in self.memory]))
        predictions = self.model(all_states)
        probabilities = predictions.gather(dim=1, index=all_actions.long().view(-1, 1)).squeeze()

        # Determine loss based on probabilities of picked actions and the rewards
        loss = - torch.sum(torch.log(probabilities) * U_theta)
        return loss


def act_in_env(n_traces: int, n_timesteps: int, param_dict: dict, bootstrapping = False, baseline = False):
    env = gym.make('CartPole-v1')               # create environment of CartPole-v1
    agent = REINFORCE_Agent(env, param_dict)    # initiate the agent

    env_scores = []                     # shows trace length over training time
    for m in range(n_traces):
        state = env.reset()             # reset environment and get initial state

        # Use the policy to collect a trace
        for t in range(n_timesteps):
            action = agent.choose_action(state)
            state_next, _ , done, _ = env.step(action)
            agent.memorize(state, action, t+1)
            state = state_next

            if done:
                env_scores.append(t+1)
                break

        # Calculate loss and update the weights of the policy
        loss = agent.loss()
        agent.optimizer.zero_grad()
        loss.backward()
        agent.optimizer.step()

        # Print the average scores over the traces
        if m % 50 == 0 and m > 0:
            print('Trace {}\tAverage Score: {:.2f}'.format(m, np.mean(env_scores[-50:-1])))

        # Reset the environment
        agent.forget()
    env.close()
    return env_scores


param_dict = {
    'alpha': 0.001,             # Learning-rate
    'gamma': 0.99               # Discount factor
}


act_in_env(n_traces=1000, n_timesteps=500, param_dict=param_dict)
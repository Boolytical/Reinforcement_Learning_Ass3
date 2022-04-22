# https://towardsdatascience.com/policy-gradient-methods-104c783251e0
import gym
import numpy as np
import torch

class policy_agent:
    def __init__(self, env, param_dict: dict):
        # Set model parameters
        self.n_states = env.observation_space.shape[0]
        self.n_actions = env.action_space.n

        self.model = self._initialize_nn()

    def _initialize_nn(self):
        print('Create a neural network with {} input nodes and {} output nodes'.format(self.n_states, self.n_actions))

        ## TO DO: Have a look at optimizing these model specifications ##
        model = torch.nn.Sequential(
            torch.nn.Linear(self.n_states, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, self.n_actions),
            torch.nn.Softmax(dim=0))

        return model    # Return the model

    def REINFORCE (self):
        ## TO DO: have a look at what this is doing and change code ##
        action_probability = self.model(torch.from_numpy(state).float())
        action = np.random.choice(np.array[0, 1], p = action_probability.data.numpy())
        return action_probability, action



def act_in_env(n_traces: int, n_timesteps: int, param_dict: dict):
    env = gym.make('CartPole-v1')            # create environment of CartPole-v1
    agent = policy_agent(env, param_dict)    # Initiate the agent
    optimizer = torch.optim.Adam(agent.model.parameters(), lr = agent.learning_rate)

    env_scores = []                     # Shows trace length over training time
    for tau in range(n_traces):
        state = env.reset()             # reset environment and get initial state
        trace = []                      # Saves the trace of the current episode

        # Use the policy to collect a trace
        for t in range(n_timesteps):
            action_probability, action = agent.REINFORCE(state)
            state_next, reward, done, _ = env.step(action)
            trace.append((state, action, t + 1))
            state = state_next

            if done:
                env_scores.append(t+1)
                break

        # Estimate the return of the trace
        R_t = torch.Tensor([r for (s, a, r) in trace]).flip(dims=(0, )) ## Is this the R_t? print it ##
        G_k = []                                            # Total return per timestep of the trace
        for i in range(len(trace)):
            new_G_k = 0
            power = 0
            for j in range(i, len(trace)):
                new_G_k = new_G_k + ((agent.gamma ** power) * R_t[j]).numpy()
                power += 1
            G_k.append(new_G_k)
        U_theta = torch.FloatTensor(G_k)                    # Expected return
        U_theta /= U_theta.max()                            ## Why is this needed? ##

        states = torch.Tensor([s for (s, a, r) in trace])
        actions = torch.Tensor([a for (s, a, r) in trace])

        predictions = agent.model(states)
        probabilities = predictions.gather(dim=1, index=actions.long().view(-1, 1)).squeeze()

        # Update the weights of the policy
        loss = - torch.sum(torch.log(probabilities) * U_theta)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if tau % 50 == 0 and tau > 0:
            print('Trace {}\tAverage Score: {:.2f}'.format(tau, np.mean(env_scores[-50:-1])))
    env.close()
    return env_scores


param_dict = {
    'alpha': 0.001,             # Learning-rate
    'gamma': 0.99               # Discount factor
}
act_in_env(500, 500, param_dict)

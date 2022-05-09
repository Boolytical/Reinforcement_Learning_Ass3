import gym
import numpy as np
import torch


class Actor_Critic_Agent:
    """ Actor-critic policy gradient used. """

    def __init__(self, env, param_dict: dict):
        """ Set parameters and initialize neural network and optimizer. """
        self.n_states = env.observation_space.shape[0]
        self.n_actions = env.action_space.n

        self.learning_rate = param_dict['alpha']
        self.n_depth = param_dict['n_depth']
        self.option = param_dict['option']
        self.NN = param_dict['NN']

        self.memory = []  # used for memorizing traces
        self.psi_values = []  # used for memorizing the psi-values

        self.model_actor = self._initialize_nn(type='actor')
        self.model_critic = self._initialize_nn(type='critic')

        self.optimizer_actor = torch.optim.Adam(self.model_actor.parameters(), lr=self.learning_rate)
        self.optimizer_critic = torch.optim.Adam(self.model_critic.parameters(), lr=self.learning_rate)

    def _initialize_nn(self, type: str = 'actor'):
        """ Initialize neural network. """
        print('Create a neural network with {} input nodes and {} output nodes'.format(self.n_states, self.n_actions))

        if isinstance(self.NN, list) and len(self.NN) == 2:
            if type == 'actor':
                model = torch.nn.Sequential(
                    torch.nn.Linear(self.n_states, self.NN[0]),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.NN[0], self.NN[1]),
                    torch.nn.Linear(self.NN[1], self.n_actions),
                    torch.nn.Softmax(dim=0))

            elif type == 'critic':
                model = torch.nn.Sequential(
                    torch.nn.Linear(self.n_states, self.NN[0]),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.NN[0], self.NN[1]),
                    torch.nn.Linear(self.NN[1], self.n_actions))

        else:
            if type == 'actor':
                model = torch.nn.Sequential(
                    torch.nn.Linear(self.n_states, self.NN),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.NN, self.n_actions),
                    torch.nn.Softmax(dim=0))

            elif type == 'critic':
                model = torch.nn.Sequential(
                    torch.nn.Linear(self.n_states, self.NN),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.NN, self.n_actions))


        return model  # return initialized model

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
        action = np.random.choice(np.arange(0, self.n_actions), p=action_probability)
        return action

    def calculate_psi(self, t, episode_length):
        """ Calculate and save the potential choices psi """
        m = min(self.n_depth, episode_length-t)
        r_t = torch.Tensor([r for (s, a, r) in self.memory]).flip(dims=(0,))
        s_t = torch.Tensor(np.array([s for (s, a, r) in self.memory]))

        if self.option == 'bootstrapping':

            s_t_depth = torch.Tensor(s_t[t + m])
            expected_return_per_action = self.model_critic(s_t_depth)  # outputs return values for action 0 and 1
            value = expected_return_per_action.mean()  # to get value of s_t+n, get highest return of output nodes

            q_val = 0  # Q_n(s_t, a_t)
            # Total return per timestep of the trace
            for k in range(m):
                q_val += r_t[t + k]

            q_val += value

            self.psi_values.append(q_val)


        elif self.option == 'baseline_subtraction':
            expected_return_per_action_t = self.model_critic(s_t[t])  # Estimated value at timestep t
            value_substract = expected_return_per_action_t.mean()

            a_val = 0
            for k in range(t, len(self.memory)):  # Compute Q-value
                a_val += r_t[k]

            a_val -= value_substract
            self.psi_values.append(a_val)

        elif self.option == 'bootstrapping_baseline':
            s_t_depth = torch.Tensor(s_t[t + m])
            expected_return_per_action = self.model_critic(s_t_depth)  # outputs return values for action 0 and 1
            value = expected_return_per_action.mean()

            expected_return_per_action_t = self.model_critic(s_t[t])  # Estimated value at timestep t
            value_substract = expected_return_per_action_t.max()

            a_val = 0
            for k in range(m):
                a_val += r_t[t + k]

            a_val = a_val + value - value_substract
            self.psi_values.append(a_val)

        else:
            raise ValueError('{} does not exist as method'.format(self.option))

    def loss(self):
        """ Return the loss for actor and critic. """
        a_t = torch.Tensor(np.array([a for (s, a, r) in self.memory]))
        s_t = torch.Tensor(np.array([s for (s, a, r) in self.memory]))

        predictions = self.model_actor(s_t)
        probabilities = predictions.gather(dim=1, index=a_t.long().view(-1, 1)).squeeze()  # get policy pi_actor(a_t|s_t) for each timestep

        ##### Update the weights of the policy
        psi = torch.Tensor(self.psi_values)
        psi.requires_grad_()

        loss_actor = - torch.sum(psi * torch.sum(torch.log(probabilities)))
        if self.option == 'bootstrapping':
            all_return_values = self.model_critic(s_t)
            value_t = torch.mean(all_return_values, 1)
            loss_critic = torch.sum(pow(psi - value_t, 2))
        else:
            loss_critic = torch.sum(pow(psi, 2))

        self.forget_psi_values()
        return loss_actor, loss_critic


def act_in_env(n_traces: int, n_timesteps: int, param_dict: dict):
    env = gym.make('CartPole-v1')  # create environment of CartPole-v1
    agent = Actor_Critic_Agent(env, param_dict)  # initiate the agent

    env_scores = []
    for e in range(n_traces):

        state = env.reset()  # reset environment and get initial state

        ##### Use current policy to collect a trace
        for t in range(n_timesteps):
            action = agent.choose_action(s=state)
            state_next, r, done, _ = env.step(action)
            agent.memorize(s=state, a=action, r=r)
            state = state_next

            if done:
                for step in range(len(agent.memory)):
                    agent.calculate_psi(t=step, episode_length=t) # if done, then calculate for every t Q_n(s_t, a_t)

                env_scores.append(t + 1)
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
            print('Epoch {}  ---   Average Score of last 50 epochs: {}'.format(e, np.mean(env_scores[-50:])))

    env.close()
    return env_scores


# Reinforcement Learning Assignment 3: REINFORCE and Actor-Critic
By Esmee Roosenmaallen, Felix Kapulla, Rosa Zwart
*** 
The provided .py files (reinforce.py, actor_critic.py, experimenter.py, Helper.py) implement agents that act in the Cartpole environment provided by Gym, OpenAI (https://gym.openai.com/envs/CartPole-v1/). The code experiments with different parameter settings and variations of the REINFORCE and actor critic agents.
***

## Descriptions of the files
***
* `Helper.py`         This file contains utility/plotter functions used in our implementation.
* `reinforce.py`      This file contains the reinforce agent class with its relevant functions as well as the function that enables the agent acting in the provided environment.   
* `actor_critic.py`  This file contains the actor critic agent class with its relevant functions as well as the function that enables the agent acting in the provided environment. This agent can adapt three different options of the actor critic algorithm, being bootstrapping, baseline subtraction and bootstrapping+baseline subtraction.   
* `experimenter.py`   This file contains the functions that interpret the arguments included in the run commands and perform the accessory experiments that in the end save the relevant plots in the same working directory.
***

## How to run the code
***
There are multiple options to run the code speficied below
`$ python experimenter.py --method >method< --option >option<`
* `>method<` can be `REINFORCE` allowing the user to fill in `>option<` with `NN` or `alpha`. The former performs the optimization of the REINFORCE agent by using several different neural network architectures. The latter performs the optimization when it comes to learning rate.
* `>method<` can be `Actor-critic` allowing the user to fill in `>option<` with `bootstrapping`, `baseline_subtraction`, `bootstrapping_baseline`. These three options perform optimization for the relevant actor critic variation by using different values for the learning rate and depth parameter.
***

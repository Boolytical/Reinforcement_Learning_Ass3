import reinforce
import numpy as np
import time
from Helper import LearningCurvePlot, smooth

def average_over_repetitions(n_repetitions, n_traces, n_timesteps, param_dict, smoothing_window):
    reward_results = np.empty([n_repetitions, n_traces]) # Result array
    now = time.time()

    for rep in range(n_repetitions):
        print(f'Repetition: {rep}')
        rewards = reinforce.act_in_env(n_traces=n_traces, n_timesteps=n_timesteps, param_dict=param_dict)
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
                'alpha': alpha, # Learning-rate
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
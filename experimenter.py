import reinforce
import actor_critic
import numpy as np
import time
from Helper import LearningCurvePlot, smooth

def average_over_repetitions(n_repetitions, n_traces, n_timesteps, param_dict, smoothing_window, method):
    reward_results = np.empty([n_repetitions, n_traces]) # Result array
    now = time.time()

    if method == 'REINFORCE':
        for rep in range(n_repetitions):
            print(f'Repetition: {rep}')
            rewards = reinforce.act_in_env(n_traces=n_traces, n_timesteps=n_timesteps, param_dict=param_dict)
            reward_results[rep] = rewards

    elif method == 'Actor-critic':
        for rep in range(n_repetitions):
            print(f'Repetition: {rep}')
            rewards = actor_critic.act_in_env(epochs = param_dict['epochs'], n_traces=n_traces,
                                              n_timesteps=n_timesteps, param_dict=param_dict)
            
            #### Below needed as the act_in_env returns a list that has length equal to epoch and every element is the average of the score per trace in that epoch
            #### We have to see what is best to do.
            reward_results = np.empty([n_repetitions, param_dict['epochs']])    # quick solution
            reward_results[rep] = rewards

    print('Running one setting takes {} minutes'.format((time.time() - now) / 60))
    learning_curve = np.mean(reward_results, axis=0)  # average over repetitions
    standard_error = np.std(reward_results, axis=0)
    learning_curve = smooth(learning_curve, smoothing_window)  # additional smoothing
    standard_error = smooth(standard_error, smoothing_window)
    return learning_curve, standard_error


def experiment(method: str):
    n_repetitions = 1
    n_traces = 10
    n_timesteps = 500
    smoothing_window = 101

    if method == 'REINFORCE':
        ### Method: REINFORCE
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

                learning_curve, standard_error = average_over_repetitions(n_repetitions, n_traces, n_timesteps, param_dict, smoothing_window, method)
                Plot.add_curve(x=np.arange(1, len(learning_curve)+1),
                            y=learning_curve,
                            std=standard_error,
                            col=colours[run],
                            label=r'REINFORCE with $\alpha$={} and $\gamma$={}'.format(alpha, gamma))
                run += 1
        Plot.save('REINFORCE.png')

    elif method == 'Actor-critic':
        ### Method: Actor-critic with bootstrapping
        n_traces = 1   # TODO: we use for this method epochs with each epoch having a 'small' number of traces
        epochs = 500
        learning_rates = [(0.001, 0.001), (0.025, 0.025), (0.01, 0.01)]
        n_depth = 10

        option = 'bootstrapping'
        colours = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]

        run = 0
        Plot = LearningCurvePlot(title=f'Method: {method} --- Results averaged of {n_repetitions} repetitions')
        for alpha in learning_rates:
                param_dict = {
                    'epochs': epochs,
                    'alpha_1': alpha[0],
                    'alpha_2': alpha[1],
                    'n_depth': n_depth,
                    'option': option
                }

                print(f'Running {method}-method with {option}: learning rate_actor ={alpha[0]} and  learning rate_critic ={alpha[1]}')

                learning_curve, standard_error = average_over_repetitions(n_repetitions, n_traces, n_timesteps, param_dict, smoothing_window, method)
                Plot.add_curve(x=np.arange(1, len(learning_curve)+1),
                            y=learning_curve,
                            std=standard_error,
                            col=colours[run],
                            label=r'Actor-critic with $\alpha_actor$={} and $\alpha_critic={}'.format(alpha[0], alpha[1]))
                run += 1
        Plot.save('Actor_critic_bootstrapping.png')


        option = 'baseline_subtraction'
        ### Method: Actor-critic with baseline subtraction
        colours = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]

        run = 0
        Plot = LearningCurvePlot(title=f'Method: {method} --- Results averaged of {n_repetitions} repetitions')
        for alpha in learning_rates:
            param_dict = {
                'epochs': epochs,
                'alpha_1': alpha[0],
                'alpha_2': alpha[1],
                'n_depth': n_depth,
                'option': option
            }

            print(f'Running {method}-method with {option}: learning rate_actor ={alpha[0]} and  learning rate_critic ={alpha[1]}')

            learning_curve, standard_error = average_over_repetitions(n_repetitions, n_traces, n_timesteps, param_dict,
                                                                    smoothing_window, method)
            Plot.add_curve(x=np.arange(1, len(learning_curve) + 1),
                        y=learning_curve,
                        std=standard_error,
                        col=colours[run],
                        label=r'Actor-critic with $\alpha_actor$={} and $\alpha_critic={}'.format(alpha[0], alpha[1]))
            run += 1
        Plot.save('Actor_critic_baseline_subtraction.png')

        option = 'bootstrapping_baseline'
        ### Method: Actor-critic with bootstrapping and baseline subtraction
        colours = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]

        run = 0
        Plot = LearningCurvePlot(title=f'Method: {method} - {option} --- Results averaged of {n_repetitions} repetitions')
        for alpha in learning_rates:
                param_dict = {
                    'epochs': epochs,
                    'alpha_1': alpha[0],
                    'alpha_2': alpha[1],
                    'n_depth': n_depth,
                    'option': option
                }

                print(f'Running {method}-method with {option}: learning rate_actor ={alpha[0]} and  learning rate_critic ={alpha[1]}')

                learning_curve, standard_error = average_over_repetitions(n_repetitions, n_traces, n_timesteps, param_dict, smoothing_window, method)
                Plot.add_curve(x=np.arange(1, len(learning_curve)+1),
                            y=learning_curve,
                            std=standard_error,
                            col=colours[run],
                            label=r'Actor-critic with $\alpha_actor$={} and $\alpha_critic={}'.format(alpha[0], alpha[1]))
                run += 1
        Plot.save('Actor_critic_both.png')

if __name__ == '__main__':
    experiment(method='Actor-critic')
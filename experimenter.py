import reinforce
import actor_critic
import numpy as np
import time
from Helper import LearningCurvePlot, smooth


def average_over_repetitions(n_repetitions, n_traces, n_timesteps, param_dict, smoothing_window, method):
    reward_results = np.empty([n_repetitions, n_traces])  # Result array
    now = time.time()

    if method == 'REINFORCE':
        for rep in range(n_repetitions):
            print(f'Repetition: {rep}')
            rewards = reinforce.act_in_env(n_traces=n_traces, n_timesteps=n_timesteps, param_dict=param_dict)
            reward_results[rep] = rewards

    elif method == 'Actor-critic':
        for rep in range(n_repetitions):
            print(f'Repetition: {rep}')
            rewards = actor_critic.act_in_env(n_traces=n_traces,
                                              n_timesteps=n_timesteps, param_dict=param_dict)

            reward_results[rep] = rewards

    print('Running one setting takes {} minutes'.format((time.time() - now) / 60))
    learning_curve = np.mean(reward_results, axis=0)  # average over repetitions
    standard_error = np.std(reward_results, axis=0)
    learning_curve = smooth(learning_curve, smoothing_window)  # additional smoothing
    standard_error = smooth(standard_error, smoothing_window)
    return learning_curve, standard_error


def experiment(method: str, option: str):
    n_repetitions = 20  # 20
    n_traces = 2000  # 2000
    n_timesteps = 500
    smoothing_window = 101

    if method == 'REINFORCE':
        ### Method: REINFORCE
        if option == 'NN':
            gamma = 0.99
            alpha = 0.001
            architecture = [64, 128, 256, [256, 128], [128, 64]]
            colours = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
                       "#bcbd22"]

            run = 0
            Plot = LearningCurvePlot(
                title=r'{} with $\alpha$={} and $\gamma$={} --- Averaged over {} repetitions'.format(method, alpha,
                                                                                                     gamma,
                                                                                                     n_repetitions))
            for NN in architecture:
                param_dict = {
                    'alpha': alpha,  # Learning-rate
                    'gamma': gamma,  # Discount factor
                    'NN': NN
                }

                print(f'Running {method}-method with learning rate={alpha} and discount rate={gamma}')

                learning_curve, standard_error = average_over_repetitions(n_repetitions, n_traces, n_timesteps,
                                                                          param_dict, smoothing_window, method)
                Plot.add_curve(x=np.arange(1, len(learning_curve) + 1),
                               y=learning_curve,
                               std=standard_error,
                               col=colours[run],
                               label=(f'REINFORCE with Neural Network Architecture = {NN}'))
                run += 1
            Plot.save('REINFORCE_NN.png')

        elif option == 'alpha':
            gamma = 0.99
            learning_rates = [0.001, 0.0025, 0.01]
            NN = [128, 64]
            colours = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
                       "#bcbd22"]

            run = 0
            Plot = LearningCurvePlot(title=f'{method} with NN {NN} --- Averaged over {n_repetitions} repetitions')
            for alpha in learning_rates:
                param_dict = {
                    'alpha': alpha,  # Learning-rate
                    'gamma': gamma,  # Discount factor
                    'NN': NN
                }

                print(f'Running {method}-method with learning rate={alpha} and discount rate={gamma}')

                learning_curve, standard_error = average_over_repetitions(n_repetitions, n_traces, n_timesteps,
                                                                          param_dict, smoothing_window, method)
                Plot.add_curve(x=np.arange(1, len(learning_curve) + 1),
                               y=learning_curve,
                               std=standard_error,
                               col=colours[run],
                               label=r'REINFORCE with $\alpha$={} and $\gamma$={}'.format(alpha, gamma))
                run += 1
            Plot.save('REINFORCE_alpha.png')

    elif method == 'Actor-critic':
        learning_rate = [0.001, 0.0025, 0.01]
        n_depth = [20, 50]
        NN = 128
        colours = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]

        if option == 'bootstrapping':
            ### Method: Actor-critic with bootstrapping
            run = 0
            Plot = LearningCurvePlot(title=f'{method} with Bootstrap \nAveraged results over {n_repetitions} repetitions')
            Plot2 = LearningCurvePlot(title=f'{method} with Bootstrap \nAveraged results over {n_repetitions} repetitions')
            for alpha in learning_rate:
                for depth in n_depth:
                    param_dict = {
                        'alpha': alpha,
                        'n_depth': depth,
                        'option': option,
                        'NN': NN
                    }

                    print(f'Running {method}-method with {option}: learning rate {alpha} and depth {depth}')

                    learning_curve, standard_error = average_over_repetitions(n_repetitions, n_traces, n_timesteps,
                                                                              param_dict, smoothing_window, method)
                    Plot.add_curve(x=np.arange(1, len(learning_curve) + 1),
                                   y=learning_curve,
                                   std=standard_error,
                                   col=colours[run],
                                   label=r'Actor-critic with $\alpha$ = {} and depth = {}'.format(alpha, depth))

                    Plot2.add_curve(x=np.arange(1, len(learning_curve) + 1),
                                    y=learning_curve,
                                    col=colours[run],
                                    label=r'Actor-critic with $\alpha$ = {} and depth = {}'.format(alpha, depth))

                    run += 1
            Plot.save('Actor_critic_bootstrapping_std.png')
            Plot2.save('Actor_critic_bootstrapping.png')

        elif option == 'baseline_subtraction':
            ### Method: Actor-critic with baseline subtraction
            run = 0
            Plot = LearningCurvePlot(title=f'{method} with Baseline Subtraction\nAveraged results over {n_repetitions} repetitions')
            Plot2 = LearningCurvePlot(title=f'{method} with Baseline Subtraction\nAveraged results over {n_repetitions} repetitions')
            for alpha in learning_rate:
                for depth in n_depth:
                    param_dict = {
                        'alpha': alpha,
                        'n_depth': np.infty,
                        'option': option,
                        'NN': NN
                    }

                    print(f'Running {method}-method with {option}: learning rate = {alpha} and depth {depth}')

                    learning_curve, standard_error = average_over_repetitions(n_repetitions, n_traces, n_timesteps,
                                                                              param_dict,
                                                                              smoothing_window, method)
                    Plot.add_curve(x=np.arange(1, len(learning_curve) + 1),
                                   y=learning_curve,
                                   std=standard_error,
                                   col=colours[run],
                                   label=r'Actor-critic with $\alpha={} and depth = {}'.format(alpha, depth))

                    Plot2.add_curve(x=np.arange(1, len(learning_curve) + 1),
                                    y=learning_curve,
                                    col=colours[run],
                                    label=r'Actor-critic with $\alpha={} and depth = {}'.format(alpha, depth))
                    run += 1
                Plot.save('Actor_critic_baseline_subtraction_std.png')
                Plot2.save('Actor_critic_baseline_subtraction.png')

        elif option == 'bootstrapping_baseline':
            ### Method: Actor-critic with bootstrapping and baseline subtraction
            run = 0
            Plot = LearningCurvePlot(title=f'{method} with Bootstrap and Baseline\nAveraged results over {n_repetitions} repetitions')
            Plot2 = LearningCurvePlot(title=f'{method} with Bootstrap and Baseline\nAveraged results over {n_repetitions} repetitions')
            for alpha in learning_rate:
                for depth in n_depth:
                    param_dict = {
                        'alpha': alpha,
                        'n_depth': depth,
                        'option': option,
                        'NN': NN
                    }

                    print(f'{method}-method with {option}: learning rate = {alpha}')

                    learning_curve, standard_error = average_over_repetitions(n_repetitions, n_traces, n_timesteps,
                                                                              param_dict, smoothing_window, method)
                    Plot.add_curve(x=np.arange(1, len(learning_curve) + 1),
                                   y=learning_curve,
                                   std=standard_error,
                                   col=colours[run],
                                   label=r'Actor-critic with $\alpha={} and depth = {}'.format(alpha, depth))

                    Plot2.add_curve(x=np.arange(1, len(learning_curve) + 1),
                                    y=learning_curve,
                                    col=colours[run],
                                    label=r'Actor-critic with $\alpha={} and depth = {}'.format(alpha, depth))
                    run += 1
            Plot.save('Actor_critic_both_std.png')
            Plot2.save('Actor_critic_both.png')


if __name__ == '__main__':
    experiment(method='Actor-critic', option='bootstrapping')
    # method: 'REINFORCE' --> option: 'NN', 'alpha'
    # method: 'Actor-critic' --> option: 'bootstrapping', 'baseline_subtraction', 'bootstrapping_baseline'
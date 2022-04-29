'''
An edited version of the learning curve plot from 'Reinforcement Learning' Assignment 1
'''

import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


class LearningCurvePlot:
    def __init__(self, title=None):
        self.fig, self.ax = plt.subplots()

        self.ax.set_xlabel('Trace')
        self.ax.set_ylabel('Reward')

        if title:
            self.ax.set_title(title)

    # Add new curve to the plot figure with given label name (optional) with rewards as y values
    def add_curve(self, x, y, col, std=None, label=None):
        if std is not None:
            if label:
                self.ax.plot(y, label=label)
                self.ax.fill_between(x=x, y1=y-std, y2=y+std, color=col, alpha=0.2)
            else:
                self.ax.plot(y)
                self.ax.fill_between(y1=y - std, y2=y + std, color=col,  alpha=0.2)
        elif label:
            self.ax.plot(y, label=label)
        else:
            self.ax.plot(y)

    # Set the upper/lower bounds of the y-axis
    def set_ylim(self, lower, upper):
        self.ax.set_ylim([lower, upper])

    # Save the plot figure given file name
    def save(self, filename):
        self.ax.legend(prop={'size': 7})
        self.fig.savefig(filename, dpi=300)


# Use Savitzky-Golayfilter for smoothing given y values
def smooth(y, window, poly=1):
    return savgol_filter(y, window, poly)
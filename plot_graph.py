import matplotlib.pyplot as plt
import numpy as np

class PltResults:
    def __init__(self):
        self.fig = plt.figure()

    def plot(self, scores):
        # add_subplot(row, column, position)
        # e.g. (2, 1, 2): 2 rows, 1 column, second row
        ax = self.fig.add_subplot(1, 1, 1)
        ax.plot(np.arange(len(scores)), scores)
        ax.set(xlabel="Episode", ylabel="Score")

        plt.show()
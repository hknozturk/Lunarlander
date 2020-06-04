import argparse

import matplotlib.pyplot as plt
import numpy as np

def plot_result(results, legends):
    for result in results:
        plt.plot(result, label={legends[results.index(result)]})
    plt.legend()
    plt.show()


if __name__ == "__main__":
    results = []
    legends = []
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_files", nargs="+", help="Choose a result (.npy) Numpy Array File to plot")
    args = parser.parse_args()

    for result_file in args.result_files:
        results.append(np.load(result_file))
        legends.append(result_file)

    plot_result(results, legends)
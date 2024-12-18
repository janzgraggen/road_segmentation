import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main(metric):
    runs = os.listdir("runs")
    plot_data = []
    for run in runs:
        if "exp" in run:
            print(f"Loading run {run}")
            file = os.listdir(f"runs/{run}")[0]
            path = f"runs/{run}/{file}"

            event_acc = EventAccumulator(path)
            event_acc.Reload()

            x = [(s.step, s.value) for s in event_acc.Scalars(metric)]
            plot_data.append(x)

    # Create a plot
    print("Creating plot")
    plt.figure()
    for data in plot_data:
        x, y = zip(*data)
        plt.plot(x, y)

    plt.title(f"Comparison of runs using metric {metric}")
    plt.legend([runs[i] for i in range(len(plot_data))])
    plt.xlabel("Global step")
    plt.ylabel(metric)
    plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--metric", type=str)
    args = ap.parse_args()
    main(args.metric)

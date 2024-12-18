import argparse
import os

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main(metric, name, prefix):
    runs = os.listdir("runs")
    runs = list(filter(lambda run: run.startswith(prefix), runs))

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
    max_values = []
    min_values = []

    for data in plot_data:
        x, y = zip(*data)
        plt.plot(x, y)
        max_values.append(max(y))
        min_values.append(min(y))

    print("\nMax values:")
    for i in range(len(max_values)):
        print(f"{i}: {max_values[i]}")

    print("\nMin values:")
    for i in range(len(min_values)):
        print(f"{i}: {min_values[i]}")

    plt.legend([runs[i] for i in range(len(plot_data))])
    plt.xlabel("Global step")
    plt.ylabel(name)
    plt.tight_layout()
    plt.savefig(f"figures/{prefix}_{metric}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--metric", type=str)
    ap.add_argument("-n", "--name", type=str)
    ap.add_argument("-p", "--prefix", type=str)

    args = ap.parse_args()
    main(args.metric, args.name, args.prefix)

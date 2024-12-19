import argparse
import os

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main(metric, name, prefix, additional, limit):
    runs = os.listdir("runs")
    runs = list(filter(lambda run: run.startswith(prefix) or run == additional, runs))
    runs = sorted(runs, key=lambda run: (len(run), run))

    plot_data = []
    for run in runs:
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
    if limit:
        plt.ylim(0.65, 0.85)

    plt.tight_layout()
    plt.savefig(f"figures/{prefix}_{metric}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--metric", type=str, default="inference_f1")
    ap.add_argument("-n", "--name", type=str, default="F1-score")
    ap.add_argument("-p", "--prefix", type=str)
    ap.add_argument("-a", "--additional", type=str, default="baseline")
    ap.add_argument("-l", "--limit", action="store_true")

    args = ap.parse_args()
    main(args.metric, args.name, args.prefix, args.additional, args.limit)

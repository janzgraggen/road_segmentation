import os

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main():
    runs = os.listdir("runs")
    runs.sort()

    data = {}
    for run in runs:
        print(f"Loading run {run}")
        file = os.listdir(f"runs/{run}")[0]
        path = f"runs/{run}/{file}"

        event_acc = EventAccumulator(path)
        event_acc.Reload()

        values = [s.value for s in event_acc.Scalars("inference_f1")]

        run_name = run.replace("exp_", "")
        run_name = run_name.replace("_", " ")

        data[run_name] = max(values)

    items = data.items()
    items = sorted(items, key=lambda item: item[1])

    SUBMISSION = "submission"

    latex = ""
    latex += "\\begin{table}[h!]\n"
    latex += "\\centering\n"
    latex += "\\begin{tabular}{l|c}\n"
    latex += "Experiment & F1 score \\\\ \n"
    latex += "\\hline\n"

    for name, value in items:
        latex += f"{name} & {value:.3f} \\\\\n"

    latex += "\\end{tabular}\n"
    latex += "\\vspace{2mm}"
    latex += "\\caption{Results of the all feature engineering steps}\n"
    latex += "\\label{tab:results}\n"
    latex += "\\end{table}"

    with open("tables/results.tex", "w") as f:
        f.write(latex)


if __name__ == "__main__":
    main()

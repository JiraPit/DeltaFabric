import os
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def parse_filename(filename):
    basename = os.path.basename(filename)
    if basename.startswith("epoch_times_single"):
        return ("single", "node_1")
    elif "dist2_node_" in basename:
        node = basename.split("dist2_node_")[1].split(".")[0]
        return ("dist2", f"node_{node}")
    elif "dist3_node_" in basename:
        node = basename.split("dist3_node_")[1].split(".")[0]
        return ("dist3", f"node_{node}")
    return None


def read_epoch_times(filepath):
    with open(filepath, "r") as f:
        content = f.read().strip()
        times = [float(t) for t in content.split(",")]
        return times


def main():
    txt_files = glob.glob("**/epoch_times*.txt", recursive=True)

    raw_data = []
    for filepath in txt_files:
        result = parse_filename(filepath)
        if result:
            run_type, node = result
            times = read_epoch_times(filepath)
            raw_data.append(
                {
                    "run_type": run_type,
                    "node": node,
                    "times": times,
                    "median_time": np.median(times),
                }
            )

    if not raw_data:
        print("No epoch_times*.txt files found!")
        return

    # Combine all epoch times per run type (for speedup calculation)
    run_times = {}
    for entry in raw_data:
        run = entry["run_type"]
        if run not in run_times:
            run_times[run] = []
        run_times[run].extend(entry["times"])

    # Calculate per-run medians
    run_medians = {run: np.median(times) for run, times in run_times.items()}

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # Top chart: Median duration per node (grouped by run type)
    df = pd.DataFrame(raw_data)
    sns.barplot(
        data=df,
        x="run_type",
        y="median_time",
        hue="node",
        ax=ax1,
        order=["single", "dist2", "dist3"],
    )
    ax1.set_title("Median Epoch Duration by Run Type and Node")
    ax1.set_xlabel("Run Type")
    ax1.set_ylabel("Median Epoch Duration (seconds)")
    ax1.legend(title="Node")
    for container in ax1.containers:
        ax1.bar_label(container, fmt="%.2fs")

    # Bottom chart: Speedup relative to single run
    if "single" in run_medians:
        single_median = run_medians["single"]
        speedup_data = []
        for run, median in run_medians.items():
            speedup = single_median / median if median > 0 else 0
            speedup_data.append({"run_type": run, "speedup": speedup})
        df_speedup = pd.DataFrame(speedup_data)

        sns.barplot(
            data=df_speedup,
            x="run_type",
            y="speedup",
            ax=ax2,
            order=["single", "dist2", "dist3"],
        )
        ax2.set_title("Speedup Relative to Single Run (Baseline = 1)")
        ax2.set_xlabel("Run Type")
        ax2.set_ylabel("Speedup")
        ax2.axhline(y=1, color="r", linestyle="--", alpha=0.5, label="Baseline")
        ax2.legend()

        for container in ax2.containers:
            ax2.bar_label(container, fmt="%.2fx")
    else:
        ax2.set_title("Speedup chart: Single run not found")
        ax2.axis("off")

    plt.tight_layout()
    plt.savefig("epoch_times_analysis.png", dpi=150)
    print("Plots saved to epoch_times_analysis.png")
    plt.show()


if __name__ == "__main__":
    main()

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt


def parse_filename(filename):
    basename = os.path.basename(filename)

    if basename.startswith("accuracy_single"):
        return ("single", None, None)

    # Parse: accuracy_dist2_node_1_alpha_0.25.txt
    name = basename.replace("accuracy_", "").replace(".txt", "")

    if "dist2_node_" in name:
        rest = name.split("dist2_node_")[1]
        parts = rest.split("_alpha_")
        node = parts[0]
        alpha = parts[1] if len(parts) > 1 else None
        return ("dist2", node, alpha)
    elif "dist3_node_" in name:
        rest = name.split("dist3_node_")[1]
        parts = rest.split("_alpha_")
        node = parts[0]
        alpha = parts[1] if len(parts) > 1 else None
        return ("dist3", node, alpha)

    return None


def read_accuracies(filepath):
    with open(filepath, "r") as f:
        content = f.read().strip()
        if content:
            return [float(a) for a in content.split(",")]
        return []


def main():
    txt_files = glob.glob("**/accuracy*.txt", recursive=True)

    all_data = []
    for filepath in txt_files:
        result = parse_filename(filepath)
        if result:
            run_type, node_id, alpha = result
            accuracies = read_accuracies(filepath)

            for epoch_idx, acc in enumerate(accuracies):
                all_data.append(
                    {
                        "run_type": run_type,
                        "node_id": node_id,
                        "alpha": alpha,
                        "epoch": epoch_idx + 1,
                        "accuracy": acc,
                    }
                )

    if not all_data:
        print("No accuracy*.txt files found!")
        return

    df = pd.DataFrame(all_data)

    # Filter out single run for the dist2/dist3 plots
    dist_df = df[df["run_type"].isin(["dist2", "dist3"])]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot for dist2
    dist2_data = dist_df[dist_df["run_type"] == "dist2"]
    if not dist2_data.empty:
        for alpha_val, group in dist2_data.groupby("alpha"):
            avg_data = group.groupby("epoch")["accuracy"].mean().reset_index()
            label = f"alpha={alpha_val}" if alpha_val else "default"
            ax1.plot(
                avg_data["epoch"],
                avg_data["accuracy"],
                marker="o",
                label=label,
                linewidth=2,
            )
        ax1.set_title("Dist2: Accuracy over Epochs")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1])
    else:
        ax1.set_title("Dist2: No data found")
        ax1.axis("off")

    # Plot for dist3
    dist3_data = dist_df[dist_df["run_type"] == "dist3"]
    if not dist3_data.empty:
        for alpha_val, group in dist3_data.groupby("alpha"):
            avg_data = group.groupby("epoch")["accuracy"].mean().reset_index()
            label = f"alpha={alpha_val}" if alpha_val else "default"
            ax2.plot(
                avg_data["epoch"],
                avg_data["accuracy"],
                marker="o",
                label=label,
                linewidth=2,
            )
        ax2.set_title("Dist3: Accuracy over Epochs")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])
    else:
        ax2.set_title("Dist3: No data found")
        ax2.axis("off")

    plt.tight_layout()
    plt.savefig("accuracy_analysis.png", dpi=150)
    print("Plot saved to accuracy_analysis.png")
    plt.show()


if __name__ == "__main__":
    main()

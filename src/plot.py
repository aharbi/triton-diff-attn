import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from pathlib import Path

plt.style.use("./results/figures/config.mplstyle")


def difference_density(filename: Path):

    df = pd.read_csv(filename)

    data = df.to_numpy()

    plt.figure(figsize=(4, 1.25))

    sns.kdeplot(data, fill=True, label="Kernel Vs. Reference")
    plt.legend()
    plt.xlabel("Absolute Difference")
    plt.ylabel("Density")
    plt.savefig("./results/figures/difference_density.pdf")


def runtime_memory_barplot(runtime_filename: Path, memory_filename: Path):

    df_runtime = pd.read_csv(runtime_filename)
    df_memory = pd.read_csv(memory_filename)

    main_columns = [
        "A (PyTorch)",
        "A (FlashAttention-2)",
        "DA (PyTorch)",
        "DA (FlashAttention-2)",
        "DA (Kernel)",
    ]
    min_columns = [col + "-min" for col in main_columns]
    max_columns = [col + "-max" for col in main_columns]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.25), sharex=True)

    x = range(int(len(df_runtime["seq_len"])))

    for main, min_col, max_col in zip(main_columns, min_columns, max_columns):
        ax1.plot(
            x,
            df_runtime[main],
            label=main,
            linestyle="dashdot",
            marker="o",
            linewidth=1,
            markersize=3,
        )
        ax1.fill_between(x, df_runtime[min_col], df_runtime[max_col], alpha=0.2)

    ax1.set_yscale("log")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"${int(val)}$" for val in df_runtime["seq_len"]], rotation=45)
    ax1.set_xlabel(r"Sequence Length ($N$)")
    ax1.set_ylabel("Runtime (ms)")
    ax1.set_title("Attention Runtime (Forward Pass)")
    ax1.legend(title="Method", fontsize="small")

    for main in main_columns:
        ax2.plot(
            x,
            df_memory[main],
            label=main,
            linestyle="dashdot",
            marker="o",
            linewidth=1,
            markersize=3,
        )

    ax2.set_yscale("log")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"${int(val)}$" for val in df_runtime["seq_len"]], rotation=45)
    ax2.set_xlabel(r"Sequence Length ($N$)")
    ax2.set_ylabel("Memory Footprint (MB)")
    ax2.set_title("Attention Memory Usage (Forward Pass)")

    plt.tight_layout()
    plt.savefig("./results/figures/runtime_memory_barplot.pdf")

    plt.show()


if __name__ == "__main__":
    difference_path = Path("./results/difference/difference.csv")
    runtime_path = Path("./results/runtime/runtime.csv")
    memorory_path = Path("./results/memory/memory.csv")

    difference_density(difference_path)
    runtime_memory_barplot(runtime_path, memorory_path)

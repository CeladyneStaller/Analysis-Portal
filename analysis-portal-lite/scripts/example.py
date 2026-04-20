"""Generates summary statistics, plots, and an Excel report from CSV data."""

from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Your scripts can import shared helpers:
# from scripts.helpers.plotting import apply_house_style


def run(input_dir: str, output_dir: str) -> dict:
    inp = Path(input_dir)
    out = Path(output_dir)
    csvs = sorted(inp.glob("*.csv"))

    if not csvs:
        return {"status": "error", "message": "No CSV files found"}

    summaries = []
    plots = []

    for csv in csvs:
        df = pd.read_csv(csv)

        # Summary stats
        desc = df.describe().T
        desc["source"] = csv.name
        summaries.append(desc)

        # Plot numeric columns
        num = df.select_dtypes(include="number").columns
        if len(num) > 0:
            ncols = min(len(num), 4)
            fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 3.5), squeeze=False)
            for i, col in enumerate(num[:4]):
                ax = axes[0][i]
                ax.plot(df[col], linewidth=0.9, color="#47b5ff")
                ax.set_title(col, fontsize=9)
                ax.tick_params(labelsize=8)
                ax.grid(True, alpha=0.15)
            fig.suptitle(csv.stem, fontsize=11, fontweight="bold")
            fig.tight_layout()
            name = f"{csv.stem}.png"
            fig.savefig(out / name, dpi=150, bbox_inches="tight")
            plt.close(fig)
            plots.append(name)

    # Excel workbook
    xl = "summary.xlsx"
    combined = pd.concat(summaries, ignore_index=True)
    with pd.ExcelWriter(out / xl, engine="openpyxl") as w:
        combined.to_excel(w, sheet_name="Summary", index=True)
        for csv in csvs:
            df = pd.read_csv(csv)
            df.to_excel(w, sheet_name=csv.stem[:31], index=False)

    return {"status": "success", "files": plots + [xl]}

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ================================================================
#                 GLOBAL STYLE & DATA PREPARATION
# ================================================================
sns.set_theme(style="whitegrid")

# Load data
df = pd.read_csv("extracted_results_longvideobench.csv")

# Extract parameters from Setting Name
df["k"] = df["Setting Name"].str.extract(r"k(\d+)")
df["alpha"] = df["Setting Name"].str.extract(r"alpha([\d.]+)").astype(float)
df["sup"] = df["Setting Name"].str.extract(r"sup([\d.]+)").astype(float)
df["method"] = df["Setting Name"].str.extract(r"(score_diff|temporal|uniform)")

# Uniform column order
METHOD_ORDER = ["score_diff", "temporal", "uniform"]

# Color palette
HEATMAP_CMAP = "RdYlGn"

# ================================================================
#                   HELPER FUNCTION FOR HEATMAPS
# ================================================================
def make_heatmap(ax, df_slice, value_col, title, ylabel="Config", center=60):
    """
    Creates a well-styled seaborn heatmap with better readability.
    """
    pivot = df_slice.pivot_table(
        values=value_col,
        index=[c for c in df_slice.columns if c in ["k", "alpha", "sup"] and c != "method"],
        columns="method",
        aggfunc="mean"
    )

    # Keep consistent column ordering
    pivot = pivot[[c for c in METHOD_ORDER if c in pivot.columns]]

    sns.heatmap(
        pivot,
        annot=True, fmt=".1f",
        cmap=HEATMAP_CMAP,
        center=center,
        vmin=center - 15,
        vmax=center + 15,
        linewidths=1.5, linecolor="white",
        square=False,
        cbar_kws={"label": "Accuracy %", "shrink": 0.8, "pad": 0.02},
        annot_kws={"fontsize": 10, "weight": "bold"},
        ax=ax
    )

    ax.set_title(title, fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel("Method", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold", labelpad=10)
    ax.tick_params(labelsize=10, width=1, length=4)

    # Readable labels with better rotation
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha="right")


# ================================================================
#                           CREATE FIGURE
# ================================================================
# Increased height for taller cells
fig = plt.figure(figsize=(22, 32))  # Taller canvas
gs = fig.add_gridspec(
    5, 3,
    hspace=0.45,     # Vertical spacing between rows
    wspace=0.35,     # Horizontal spacing between columns
    left=0.08,
    right=0.95,
    top=0.96,
    bottom=0.03
)

fig.suptitle(
    "LongVideoBench Performance Heatmap\nMethod × Parameter Analysis",
    fontsize=22, fontweight="bold", y=0.98
)

# ================================================================
#                         ROW 1 – OVERALL
# ================================================================
settings_row1 = [
    ("Overall Accuracy", "Overall (All k)", None),
    ("Overall Accuracy", "Overall (k = 8)", "8"),
    ("Overall Accuracy", "Overall (k = 16)", "16"),
]

for col, (metric, title, k_value) in enumerate(settings_row1):
    ax = fig.add_subplot(gs[0, col])

    df_slice = df.copy() if k_value is None else df[df["k"] == k_value].copy()

    # Drop "k" from index when fixed
    ylabel = "k, α, sup" if k_value is None else "α, sup"

    make_heatmap(ax, df_slice, metric, title, ylabel=ylabel, center=60)


# ================================================================
#                  ROWS 2–5 – PER DURATION
# ================================================================
duration_metrics = [
    ("15s %",   "15 seconds"),
    ("60s %",   "60 seconds"),
    ("600s %",  "600s (10 min)"),
    ("3600s %", "3600s (1 hr)")
]

for row, (metric, label) in enumerate(duration_metrics, start=1):

    # Column 0 — All k
    ax = fig.add_subplot(gs[row, 0])
    make_heatmap(ax, df, metric, f"{label} (All k)", ylabel="k, α, sup", center=58)

    # Column 1 — k = 8
    ax = fig.add_subplot(gs[row, 1])
    make_heatmap(ax, df[df["k"] == "8"], metric, f"{label} (k = 8)", ylabel="α, sup", center=58)

    # Column 2 — k = 16
    ax = fig.add_subplot(gs[row, 2])
    make_heatmap(ax, df[df["k"] == "16"], metric, f"{label} (k = 16)", ylabel="α, sup", center=58)

# ================================================================
#                        SAVE OUTPUT
# ================================================================
OUTFILE = "longvideobench_heatmap_poster.png"
plt.savefig(OUTFILE, dpi=150, bbox_inches="tight", pad_inches=0.3)
plt.show()

print("=" * 90)
print(" ✔ READABLE HEATMAP GENERATED SUCCESSFULLY!")
print("=" * 90)
print(f"File: {OUTFILE}")
print(f"Canvas Size: 22 × 32 inches")
print(f"Resolution: 3300 × 4800 px (at 150 DPI)")
print("=" * 90)
print("Layout:")
print("Row 1:  Overall Accuracy        → [All k | k=8 | k=16]")
print("Row 2:  15s Accuracy             → [All k | k=8 | k=16]")
print("Row 3:  60s Accuracy             → [All k | k=8 | k=16]")
print("Row 4:  600s Accuracy            → [All k | k=8 | k=16]")
print("Row 5:  3600s Accuracy           → [All k | k=8 | k=16]")
print("=" * 90)
print("\n🎨 IMPROVEMENTS:")
print("  • Increased canvas height: 24 → 32 inches (taller cells)")
print("  • Slightly wider: 20 → 22 inches")
print("  • Optimized spacing for better cell height")
print("  • Increased font sizes slightly for better visibility")
print("  • Better aspect ratio for heatmap cells")
print("=" * 90)
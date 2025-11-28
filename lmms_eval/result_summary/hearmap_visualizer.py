# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np

# # Load data
# df = pd.read_csv('extracted_results_videomme_300.csv')

# # Extract parameters from setting name
# df['k'] = df['Setting Name'].str.extract(r'k(\d+)')
# df['alpha'] = df['Setting Name'].str.extract(r'alpha([\d.]+)')
# df['sup'] = df['Setting Name'].str.extract(r'sup([\d.]+)')
# df['method'] = df['Setting Name'].str.extract(r'(score_diff|temporal)')

# # Create heatmap for each k value
# fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# for idx, k_val in enumerate(['8', '16']):
#     df_k = df[df['k'] == k_val].copy()
    
#     # Create pivot table
#     pivot = df_k.pivot_table(
#         values='Overall Accuracy',
#         index=['alpha', 'sup'],
#         columns='method',
#         aggfunc='mean'
#     )
    
#     sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn', 
#                 center=60, vmin=50, vmax=70, ax=axes[idx], cbar_kws={'label': 'Accuracy %'})
#     axes[idx].set_title(f'Overall Accuracy (k={k_val})', fontsize=14, fontweight='bold')
#     axes[idx].set_xlabel('Method', fontsize=12)
#     axes[idx].set_ylabel('Alpha, Sup', fontsize=12)

# plt.tight_layout()
# plt.savefig('heatmap_overall.png', dpi=300, bbox_inches='tight')
# plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ================================================================
#                 GLOBAL STYLE & DATA PREPARATION
# ================================================================
sns.set_theme(style="whitegrid")

# Load data
df = pd.read_csv('extracted_results_videomme_300.csv')

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
# Increased height for 4 rows (Overall + Short + Medium + Long)
fig = plt.figure(figsize=(22, 32))  # Taller canvas for 4 rows
gs = fig.add_gridspec(
    4, 3,  # 4 rows, 3 columns
    hspace=0.45,     # Vertical spacing between rows
    wspace=0.35,     # Horizontal spacing between columns
    left=0.08,
    right=0.95,
    top=0.96,
    bottom=0.03
)

fig.suptitle(
    "VideoMME Performance Heatmap\nMethod × Parameter Analysis",
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
#                  ROWS 2–4 – SHORT, MEDIUM, LONG
# ================================================================
duration_metrics = [
    ("Short %", "Short Videos"),      # ✅ Changed from "Short" to "Short %"
    ("Medium %", "Medium Videos"),    # ✅ Changed from "Medium" to "Medium %"
    ("Long %", "Long Videos")         # ✅ Changed from "Long" to "Long %"
]

for row, (metric, label) in enumerate(duration_metrics, start=1):

    # Column 0 — All k
    ax = fig.add_subplot(gs[row, 0])
    make_heatmap(ax, df, metric, f"{label} (All k)", ylabel="k, α, sup", center=60)

    # Column 1 — k = 8
    ax = fig.add_subplot(gs[row, 1])
    make_heatmap(ax, df[df["k"] == "8"], metric, f"{label} (k = 8)", ylabel="α, sup", center=60)

    # Column 2 — k = 16
    ax = fig.add_subplot(gs[row, 2])
    make_heatmap(ax, df[df["k"] == "16"], metric, f"{label} (k = 16)", ylabel="α, sup", center=60)

# ================================================================
#                        SAVE OUTPUT
# ================================================================
OUTFILE = "videomme_heatmap_poster.png"
plt.savefig(OUTFILE, dpi=150, bbox_inches="tight", pad_inches=0.3)
plt.show()

print("=" * 90)
print(" ✔ VIDEOMME HEATMAP GENERATED SUCCESSFULLY!")
print("=" * 90)
print(f"File: {OUTFILE}")
print(f"Canvas Size: 22 × 32 inches")
print(f"Resolution: 3300 × 4800 px (at 150 DPI)")
print("=" * 90)
print("Layout:")
print("Row 1:  Overall Accuracy        → [All k | k=8 | k=16]")
print("Row 2:  Short Videos            → [All k | k=8 | k=16]")
print("Row 3:  Medium Videos           → [All k | k=8 | k=16]")
print("Row 4:  Long Videos             → [All k | k=8 | k=16]")
print("=" * 90)
print("\n🎨 STYLE CONSISTENCY:")
print("  • Matches LongVideoBench styling perfectly")
print("  • Same 4-row layout structure")
print("  • Same color scheme (RdYlGn)")
print("  • Same font sizes and spacing")
print("  • Same annotation style")
print("  • Consistent method ordering")
print("=" * 90)
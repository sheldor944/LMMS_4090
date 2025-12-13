import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# --- 1. Data Loading ---
CSV_FILE_NAME = 'extracted_results_videomme_TMAS.csv'
df = pd.read_csv(CSV_FILE_NAME)

# --- Create output folder based on CSV name ---
folder_name = os.path.splitext(CSV_FILE_NAME)[0]  # Remove .csv extension
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
    print(f"✓ Created folder: {folder_name}/")
else:
    print(f"✓ Using existing folder: {folder_name}/")

# Check actual columns
print("Original columns:", df.columns.tolist())

# Use actual column names from CSV (no renaming needed if they match)
# The CSV has: Setting Name, Overall Accuracy, Short %, Medium %, Long %, Short (correct/total), Medium (correct/total), Long (correct/total)

# Rename columns to remove spaces and special characters for easier access
df.columns = [
    'Setting_Name', 'Overall_Accuracy', 'Short_pct', 'Medium_pct', 'Long_pct',
    'Short_correct_total', 'Medium_correct_total', 'Long_correct_total'
]

# Define metric columns (only the percentage ones we care about)
metric_columns = ['Overall_Accuracy', 'Short_pct', 'Medium_pct', 'Long_pct']

# Create a simplified Setting_ID (e.g., S1, S2, ...) for readable bar chart labels
df['Setting_ID'] = [f'S{i+1}' for i in range(len(df))]

# --- 2. Extract meaningful parameters from Setting_Name for analysis ---
df['method'] = df['Setting_Name'].str.extract(r'(score_diff|temporal|uniform)')

# Fix for iteration extraction - handle NaN values
iteration_extracted = df['Setting_Name'].str.extract(r'iter(\d+)')
df['iteration'] = pd.to_numeric(iteration_extracted[0], errors='coerce').fillna(0).astype(int)

df['k'] = df['Setting_Name'].str.extract(r'_k(\d+)_')

# Fix for alpha extraction - handle NaN values
alpha_extracted = df['Setting_Name'].str.extract(r'alpha([\d.]+)')
df['alpha'] = pd.to_numeric(alpha_extracted[0], errors='coerce')

# Fix for short_sup extraction - handle NaN values
short_sup_extracted = df['Setting_Name'].str.extract(r'short_([\d.]+)')
df['short_sup'] = pd.to_numeric(short_sup_extracted[0], errors='coerce')

# Fix for med_sup extraction - handle NaN values
med_sup_extracted = df['Setting_Name'].str.extract(r'med_([\d.]+)')
df['med_sup'] = pd.to_numeric(med_sup_extracted[0], errors='coerce')

# Fix for long_sup extraction - handle NaN values
long_sup_extracted = df['Setting_Name'].str.extract(r'long_([\d.]+)')
df['long_sup'] = pd.to_numeric(long_sup_extracted[0], errors='coerce')

# Create adaptive_config, handling NaN values
df['adaptive_config'] = df.apply(
    lambda x: f"s{x['short_sup']}_m{x['med_sup']}_l{x['long_sup']}" 
    if pd.notna(x['short_sup']) and pd.notna(x['med_sup']) and pd.notna(x['long_sup'])
    else 'unknown', 
    axis=1
)

print("\nExtracted parameters:")
print(df[['Setting_Name', 'method', 'iteration', 'adaptive_config']].head(10))

# --- 3. Heatmap Visualization (All metrics vs. Settings) ---

# Prepare data for heatmap: set 'Setting_Name' as index and select metric columns
heatmap_data = df.set_index('Setting_Name')[metric_columns]

plt.figure(figsize=(10, max(12, len(df) * 0.5)))
sns.heatmap(
    heatmap_data,
    annot=True,
    fmt=".2f",
    cmap="YlGnBu",
    linewidths=.5,
    linecolor='black',
    cbar_kws={'label': 'Accuracy (%)'}
)
plt.title('VideoMME: Heatmap of All Accuracy Metrics by Setting Name', fontsize=16)
plt.ylabel('Setting Name (Full)', fontsize=14)
plt.yticks(rotation=0, fontsize=7)
plt.xlabel('Metric', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.tight_layout()
plt.savefig(f'{folder_name}/videomme_heatmap_all_metrics.png', dpi=150)
plt.close()
print(f"✓ Saved: {folder_name}/videomme_heatmap_all_metrics.png")


# --- 4. Bar Chart Visualization (Overall Accuracy vs. Simplified Setting ID) ---

# Sort the data by Overall Accuracy for a clearer, ranked bar chart
bar_data = df.sort_values(by='Overall_Accuracy', ascending=False)

plt.figure(figsize=(10, max(8, len(df) * 0.4)))
sns.barplot(
    x='Overall_Accuracy',
    y='Setting_ID',
    data=bar_data,
    palette='viridis'
)
plt.title('VideoMME: Overall Accuracy by Setting ID (Ranked)', fontsize=16)
plt.xlabel('Overall Accuracy (%)', fontsize=14)
plt.ylabel('Setting ID', fontsize=14)

# Add labels for the exact values on the bars
for index, row in bar_data.reset_index().iterrows():
    plt.text(row.Overall_Accuracy + 0.1, index, f'{row.Overall_Accuracy:.2f}', 
             color='black', ha="left", va="center", fontsize=8)

plt.xlim(bar_data['Overall_Accuracy'].min() - 0.5, bar_data['Overall_Accuracy'].max() + 1)
plt.grid(axis='x', linestyle='--')
plt.tight_layout()
plt.savefig(f'{folder_name}/videomme_barchart_overall_accuracy.png', dpi=150)
plt.close()
print(f"✓ Saved: {folder_name}/videomme_barchart_overall_accuracy.png")


# --- 5. Grouped Bar Chart by Method (Averaged across iterations) ---
# Only create if method data exists
if df['method'].notna().any():
    method_avg = df.groupby('method')[metric_columns].mean().reset_index()
    method_melted = method_avg.melt(id_vars='method', var_name='Metric', value_name='Accuracy')

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x='Metric',
        y='Accuracy',
        hue='method',
        data=method_melted,
        palette='Set2'
    )
    plt.title('VideoMME: Average Accuracy by Method', fontsize=16)
    plt.xlabel('Metric', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.legend(title='Method', fontsize=10)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(50, 80)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{folder_name}/videomme_barchart_by_method.png', dpi=150)
    plt.close()
    print(f"✓ Saved: {folder_name}/videomme_barchart_by_method.png")
else:
    print("⚠ Skipped: videomme_barchart_by_method.png (no method data found)")


# --- 6. Grouped Bar Chart by Adaptive Config ---
# Only create if adaptive_config data exists (excluding 'unknown')
valid_configs = df[df['adaptive_config'] != 'unknown']
if len(valid_configs) > 0:
    config_avg = valid_configs.groupby('adaptive_config')[metric_columns].mean().reset_index()
    config_melted = config_avg.melt(id_vars='adaptive_config', var_name='Metric', value_name='Accuracy')

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x='Metric',
        y='Accuracy',
        hue='adaptive_config',
        data=config_melted,
        palette='Set1'
    )
    plt.title('VideoMME: Average Accuracy by Adaptive Config', fontsize=16)
    plt.xlabel('Metric', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.legend(title='Adaptive Config', fontsize=9)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(50, 80)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{folder_name}/videomme_barchart_by_config.png', dpi=150)
    plt.close()
    print(f"✓ Saved: {folder_name}/videomme_barchart_by_config.png")
else:
    print("⚠ Skipped: videomme_barchart_by_config.png (no adaptive config data found)")


# --- 7. Heatmap: Method × Adaptive Config (Overall Accuracy) ---
# Only create if both method and config data exist
if df['method'].notna().any() and len(valid_configs) > 0:
    pivot_overall = valid_configs.pivot_table(
        values='Overall_Accuracy',
        index='adaptive_config',
        columns='method',
        aggfunc='mean'
    )

    if not pivot_overall.empty:
        plt.figure(figsize=(8, 5))
        sns.heatmap(
            pivot_overall,
            annot=True,
            fmt=".2f",
            cmap="RdYlGn",
            center=pivot_overall.values.mean(),
            linewidths=1,
            linecolor='white',
            cbar_kws={'label': 'Overall Accuracy (%)'}
        )
        plt.title('VideoMME: Overall Accuracy\n(Method × Adaptive Config)', fontsize=14)
        plt.xlabel('Method', fontsize=12)
        plt.ylabel('Adaptive Config', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{folder_name}/videomme_heatmap_method_config.png', dpi=150)
        plt.close()
        print(f"✓ Saved: {folder_name}/videomme_heatmap_method_config.png")
    else:
        print("⚠ Skipped: videomme_heatmap_method_config.png (no valid pivot data)")
else:
    print("⚠ Skipped: videomme_heatmap_method_config.png (insufficient data)")


# --- 8. Multi-panel Heatmap: Each Duration Category ---
if df['method'].notna().any() and len(valid_configs) > 0:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    duration_metrics = [('Short_pct', 'Short'), ('Medium_pct', 'Medium'), ('Long_pct', 'Long')]

    for ax, (metric, label) in zip(axes, duration_metrics):
        pivot = valid_configs.pivot_table(
            values=metric,
            index='adaptive_config',
            columns='method',
            aggfunc='mean'
        )
        
        if not pivot.empty:
            sns.heatmap(
                pivot,
                annot=True,
                fmt=".1f",
                cmap="RdYlGn",
                center=pivot.values.mean(),
                linewidths=1,
                linecolor='white',
                cbar_kws={'label': 'Accuracy (%)'},
                ax=ax
            )
            ax.set_title(f'{label} Duration Accuracy', fontsize=12, fontweight='bold')
            ax.set_xlabel('Method', fontsize=10)
            ax.set_ylabel('Adaptive Config', fontsize=10)

    plt.suptitle('VideoMME: Accuracy by Duration Category', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{folder_name}/videomme_heatmap_by_duration.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {folder_name}/videomme_heatmap_by_duration.png")
else:
    print("⚠ Skipped: videomme_heatmap_by_duration.png (insufficient data)")


# --- 9. Line Plot: Performance Across Iterations ---
# Only create if iteration data exists
if df['iteration'].max() > 0:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for ax, metric in zip(axes, metric_columns):
        for method in df['method'].dropna().unique():
            for config in df['adaptive_config'].dropna().unique():
                if config != 'unknown':
                    subset = df[(df['method'] == method) & (df['adaptive_config'] == config)]
                    subset = subset.sort_values('iteration')
                    if len(subset) > 0:
                        label = f"{method[:6]}_{config}"  # Shortened label
                        ax.plot(subset['iteration'], subset[metric], marker='o', label=label, alpha=0.8)
        
        ax.set_title(f'{metric.replace("_", " ")}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Iteration', fontsize=10)
        ax.set_ylabel('Accuracy (%)', fontsize=10)
        ax.set_xticks(df['iteration'].dropna().unique())
        ax.grid(True, linestyle='--', alpha=0.5)

    # Single legend for all subplots
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='center right', fontsize=7, bbox_to_anchor=(1.18, 0.5))

    plt.suptitle('VideoMME: Performance Across Iterations', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{folder_name}/videomme_lineplot_iterations.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {folder_name}/videomme_lineplot_iterations.png")
else:
    print("⚠ Skipped: videomme_lineplot_iterations.png (no iteration data found)")


# --- 10. Create Setting ID Legend/Mapping ---

legend_df = df[['Setting_ID', 'Setting_Name', 'method', 'adaptive_config', 'iteration']].copy()
legend_df.to_csv(f'{folder_name}/videomme_setting_id_legend.csv', index=False)
print(f"✓ Saved: {folder_name}/videomme_setting_id_legend.csv")


# --- 11. Summary Statistics Table ---
print("\n" + "=" * 90)
print(" VideoMME VISUALIZATION SUMMARY")
print("=" * 90)

print(f"\n📁 Output Folder: {folder_name}/")
print("\n📊 Files Generated:")
print("  1. videomme_heatmap_all_metrics.png       - Full heatmap of all settings × metrics")
print("  2. videomme_barchart_overall_accuracy.png - Ranked bar chart of overall accuracy")
print("  3. videomme_barchart_by_method.png        - Grouped bar chart by method")
print("  4. videomme_barchart_by_config.png        - Grouped bar chart by adaptive config")
print("  5. videomme_heatmap_method_config.png     - Heatmap: Method × Config for Overall")
print("  6. videomme_heatmap_by_duration.png       - Multi-panel heatmap by duration")
print("  7. videomme_lineplot_iterations.png       - Line plots showing iteration trends")
print("  8. videomme_setting_id_legend.csv         - Mapping of Setting IDs to full names")

print("\n📈 Summary Statistics (Averaged across iterations):")
print("-" * 60)
if df['method'].notna().any() and len(valid_configs) > 0:
    summary = valid_configs.groupby(['method', 'adaptive_config'])[metric_columns].mean().round(2)
    print(summary.to_string())
else:
    print("Summary statistics not available (insufficient parameter data)")

print("\n📊 Best Configurations:")
print("-" * 60)
best_overall_idx = df['Overall_Accuracy'].idxmax()
best_overall = df.loc[best_overall_idx]
print(f"  Best Overall:  {best_overall['Setting_Name']}")
print(f"                 Accuracy: {best_overall['Overall_Accuracy']:.2f}%")

best_short_idx = df['Short_pct'].idxmax()
best_short = df.loc[best_short_idx]
print(f"  Best Short:    {best_short['Setting_Name']}")
print(f"                 Accuracy: {best_short['Short_pct']:.2f}%")

best_medium_idx = df['Medium_pct'].idxmax()
best_medium = df.loc[best_medium_idx]
print(f"  Best Medium:   {best_medium['Setting_Name']}")
print(f"                 Accuracy: {best_medium['Medium_pct']:.2f}%")

best_long_idx = df['Long_pct'].idxmax()
best_long = df.loc[best_long_idx]
print(f"  Best Long:     {best_long['Setting_Name']}")
print(f"                 Accuracy: {best_long['Long_pct']:.2f}%")

print("\n📋 Setting ID Legend:")
print("-" * 60)
for _, row in legend_df.iterrows():
    print(f"  {row['Setting_ID']}: {row['method']} | {row['adaptive_config']} | iter{row['iteration']}")

print("\n" + "=" * 90)
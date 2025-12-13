import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# --- 1. Data Loading ---
CSV_FILE_NAME = 'extracted_results_longvideobench_TMAS.csv'
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

# Rename columns for easier access and cleaner plot labels
new_columns = ['Setting_Name', 'Overall_Accuracy', '15s_pct', '60s_pct', '600s_pct', '3600s_pct']

# Ensure we have the right number of columns
if len(df.columns) == len(new_columns):
    df.columns = new_columns
else:
    print(f"⚠ Warning: Expected {len(new_columns)} columns but found {len(df.columns)}")
    print(f"  Using original column names: {df.columns.tolist()}")
    # Adjust new_columns to match actual columns
    new_columns = df.columns.tolist()

# Create a simplified Setting_ID (e.g., S1, S2, ...) for readable bar chart labels
df['Setting_ID'] = [f'S{i+1}' for i in range(len(df))]

# --- 2. Heatmap Visualization (All metrics vs. Settings) ---

# Get metric columns (all except Setting_Name and Setting_ID)
metric_columns = [col for col in new_columns if col not in ['Setting_Name', 'Setting_ID']]

# Prepare data for heatmap: set 'Setting_Name' as index and select metric columns
heatmap_data = df.set_index('Setting_Name')[metric_columns]

plt.figure(figsize=(12, max(12, len(df) * 0.5)))
sns.heatmap(
    heatmap_data,
    annot=True,
    fmt=".2f",
    cmap="YlGnBu",
    linewidths=.5,
    linecolor='black',
    cbar_kws={'label': 'Accuracy (%)'}
)
plt.title('Heatmap of All Accuracy Metrics by Setting Name', fontsize=16)
plt.ylabel('Setting Name (Full)', fontsize=14)
plt.yticks(rotation=0, fontsize=8)  # Keep original rotation for long names
plt.xlabel('Metric', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.tight_layout()
plt.savefig(f'{folder_name}/heatmap_all_metrics.png', dpi=150)
plt.close()
print(f"✓ Saved: {folder_name}/heatmap_all_metrics.png")


# --- 3. Bar Chart Visualization (Overall Accuracy vs. Simplified Setting ID) ---

# Determine which column to use for overall accuracy
overall_col = 'Overall_Accuracy' if 'Overall_Accuracy' in df.columns else df.columns[1]

# Sort the data by Overall Accuracy for a clearer, ranked bar chart
bar_data = df.sort_values(by=overall_col, ascending=False)

plt.figure(figsize=(10, max(8, len(df) * 0.4)))
sns.barplot(
    x=overall_col,
    y='Setting_ID',
    data=bar_data,
    palette='viridis'
)
plt.title('Overall Accuracy by Simplified Setting ID (Ranked)', fontsize=16)
plt.xlabel('Overall Accuracy (%)', fontsize=14)
plt.ylabel('Setting ID', fontsize=14)

# Add labels for the exact values on the bars
for index, row in bar_data.reset_index().iterrows():
    accuracy_val = row[overall_col]
    if pd.notna(accuracy_val):  # Only add label if value is not NaN
        plt.text(accuracy_val + 0.1, index, f'{accuracy_val:.2f}', 
                 color='black', ha="left", va="center", fontsize=8)

# Set x-axis limits with NaN handling
valid_accuracies = bar_data[overall_col].dropna()
if len(valid_accuracies) > 0:
    plt.xlim(valid_accuracies.min() - 0.5, valid_accuracies.max() + 1)

plt.grid(axis='x', linestyle='--')
plt.tight_layout()
plt.savefig(f'{folder_name}/barchart_overall_accuracy.png', dpi=150)
plt.close()
print(f"✓ Saved: {folder_name}/barchart_overall_accuracy.png")


# --- 4. Create Setting ID Legend/Mapping ---
legend_df = df[['Setting_ID', 'Setting_Name']].copy()
legend_df.to_csv(f'{folder_name}/setting_id_legend.csv', index=False)
print(f"✓ Saved: {folder_name}/setting_id_legend.csv")


# --- 5. Summary Statistics ---
print("\n" + "=" * 90)
print(" LONGVIDEOBENCH VISUALIZATION SUMMARY")
print("=" * 90)

print(f"\n📁 Output Folder: {folder_name}/")
print("\n📊 Files Generated:")
print(f"  1. heatmap_all_metrics.png        - Heatmap of all settings × metrics")
print(f"  2. barchart_overall_accuracy.png  - Ranked bar chart of overall accuracy")
print(f"  3. setting_id_legend.csv          - Mapping of Setting IDs to full names")

print("\n📈 Summary Statistics:")
print("-" * 60)
for col in metric_columns:
    if col in df.columns:
        col_data = df[col].dropna()
        if len(col_data) > 0:
            print(f"{col:20s}: Mean={col_data.mean():.2f}%, "
                  f"Min={col_data.min():.2f}%, Max={col_data.max():.2f}%")

print("\n📊 Best Configurations:")
print("-" * 60)

# Best Overall
if overall_col in df.columns:
    best_overall_idx = df[overall_col].idxmax()
    if pd.notna(best_overall_idx):
        best_overall = df.loc[best_overall_idx]
        print(f"  Best Overall:  {best_overall['Setting_Name']}")
        print(f"                 Accuracy: {best_overall[overall_col]:.2f}%")

# Best for each duration
duration_cols = [col for col in metric_columns if col != overall_col]
for col in duration_cols:
    if col in df.columns:
        best_idx = df[col].idxmax()
        if pd.notna(best_idx):
            best = df.loc[best_idx]
            duration_label = col.replace('_pct', '').replace('_', ' ')
            print(f"  Best {duration_label:8s}: {best['Setting_Name']}")
            print(f"                 Accuracy: {best[col]:.2f}%")

print("\n📋 Setting ID Legend:")
print("-" * 60)
for _, row in legend_df.head(10).iterrows():  # Show first 10
    print(f"  {row['Setting_ID']}: {row['Setting_Name']}")
if len(legend_df) > 10:
    print(f"  ... and {len(legend_df) - 10} more (see {folder_name}/setting_id_legend.csv)")

print("\n" + "=" * 90)
print(f"✔ All visualizations generated successfully in '{folder_name}/' folder!")
print("=" * 90)
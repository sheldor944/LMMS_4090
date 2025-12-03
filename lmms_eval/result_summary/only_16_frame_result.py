import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- 1. Data Loading ---
CSV_FILE_NAME = 'extracted_results_longvideobench.csv'
df = pd.read_csv(CSV_FILE_NAME)

# Rename columns for easier access and cleaner plot labels
new_columns = ['Setting_Name', 'Overall_Accuracy', '15s_pct', '60s_pct', '600s_pct', '3600s_pct']
df.columns = new_columns

# Create a simplified Setting_ID (e.g., S1, S2, ...) for readable bar chart labels
df['Setting_ID'] = [f'S{i+1}' for i in range(len(df))]

# --- 2. Heatmap Visualization (All metrics vs. Settings) ---

# Prepare data for heatmap: set 'Setting_Name' as index and select metric columns
heatmap_data = df.set_index('Setting_Name')[new_columns[1:]]

plt.figure(figsize=(12, 12))
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
plt.yticks(rotation=0, fontsize=8) # Keep original rotation for long names
plt.xlabel('Metric', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.tight_layout()
plt.savefig('heatmap_all_metrics.png')
plt.close()


# --- 3. Bar Chart Visualization (Overall Accuracy vs. Simplified Setting ID) ---

# Sort the data by Overall Accuracy for a clearer, ranked bar chart
bar_data = df.sort_values(by='Overall_Accuracy', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(
    x='Overall_Accuracy',
    y='Setting_ID',
    data=bar_data,
    palette='viridis'
)
plt.title('Overall Accuracy by Simplified Setting ID (Ranked)', fontsize=16)
plt.xlabel('Overall Accuracy (%)', fontsize=14)
plt.ylabel('Setting ID', fontsize=14)

# Add labels for the exact values on the bars
for index, row in bar_data.reset_index().iterrows():
    plt.text(row.Overall_Accuracy + 0.1, index, f'{row.Overall_Accuracy:.2f}', 
             color='black', ha="left", va="center", fontsize=8)

plt.xlim(bar_data['Overall_Accuracy'].min() - 0.5, bar_data['Overall_Accuracy'].max() + 1)
plt.grid(axis='x', linestyle='--')
plt.tight_layout()
plt.savefig('barchart_overall_accuracy.png')
plt.close()

print(f"Two visualizations generated: 'heatmap_all_metrics.png' and 'barchart_overall_accuracy.png' from {CSV_FILE_NAME}.")
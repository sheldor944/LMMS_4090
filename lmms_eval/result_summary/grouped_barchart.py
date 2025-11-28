import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('extracted_results_videomme_300.csv')
# Get top 10 configurations by overall accuracy
top_configs = df.nlargest(10, 'Overall Accuracy').copy()

# Prepare data
settings = [s.replace('selected_dbfp_videomme_blip_', '').replace('selected_videomme_frames_', '') 
            for s in top_configs['Setting Name']]
overall = top_configs['Overall Accuracy'].values
short = top_configs['Short %'].values
medium = top_configs['Medium %'].values
long = top_configs['Long %'].values

x = np.arange(len(settings))
width = 0.2

fig, ax = plt.subplots(figsize=(18, 7))

# Plot bars
bars0 = ax.bar(x - 1.5*width, overall, width, label='Overall', color='#3498db', alpha=0.9, edgecolor='black', linewidth=1.5)
bars1 = ax.bar(x - 0.5*width, short, width, label='Short', color='#2ecc71', alpha=0.8)
bars2 = ax.bar(x + 0.5*width, medium, width, label='Medium', color='#f39c12', alpha=0.8)
bars3 = ax.bar(x + 1.5*width, long, width, label='Long', color='#e74c3c', alpha=0.8)

ax.set_xlabel('Configuration', fontsize=13, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
ax.set_title('Top 10 Configurations: Overall and Duration-wise Accuracy (Sorted by Overall)', fontsize=15, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(settings, rotation=45, ha='right', fontsize=10)
ax.legend(fontsize=12, loc='upper right')
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim([0, 100])

# Add value labels on bars
for bars in [bars0, bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=7, fontweight='bold')

# Add a horizontal line for average overall accuracy
avg_overall = overall.mean()
ax.axhline(y=avg_overall, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Avg Overall: {avg_overall:.1f}%')

plt.tight_layout()
plt.savefig('top10_duration_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Print the rankings
print("\nTop 10 Configurations (Sorted by Overall Accuracy):")
print("="*80)
for idx, row in top_configs.iterrows():
    setting = row['Setting Name'].replace('selected_dbfp_videomme_blip_', '').replace('selected_videomme_frames_', '')
    print(f"{top_configs.index.get_loc(idx)+1:2d}. {setting:50s} | Overall: {row['Overall Accuracy']:5.2f}% | Short: {row['Short %']:5.1f}% | Medium: {row['Medium %']:5.1f}% | Long: {row['Long %']:5.1f}%")

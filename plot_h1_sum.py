import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load features and labels
try:
    X_val = np.load('results/X_val.npy')
except:
    print("Could not load X_val")
    exit(1)
val_data = pd.read_csv('small_gpt_web/valid_5k.csv').head(500)

# Feature extraction logic
LAYER = 9
HEAD = 7
THRESHOLDS_LEN = 9
TOPOLOGICAL_SIZE = 4 * THRESHOLDS_LEN
feature_per_head = TOPOLOGICAL_SIZE + 16 + 5 * THRESHOLDS_LEN
# H1 sum is index 1 of the H1 barcode features (offset 8 from H0)
head_idx = LAYER * 12 + HEAD
h1_sum_col = head_idx * feature_per_head + TOPOLOGICAL_SIZE + 8 + 1

# Extract values into a dataframe for seaborn
df = pd.DataFrame({
    'H1_Sum': X_val[:, h1_sum_col],
    'Label': val_data['label']
})

# Plot
plt.figure(figsize=(8, 6))
sns.kdeplot(data=df[df['Label'] == 'generated'], x='H1_Sum', label='Generated (GPT-2 Small)', fill=True, color='#f29243', alpha=0.5, linewidth=2)
sns.kdeplot(data=df[df['Label'] == 'natural'], x='H1_Sum', label='Natural (WebText)', fill=True, color='#5a7eb8', alpha=0.5, linewidth=2)

plt.title(f'Layer {LAYER}, Head {HEAD}', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Sum of length of bars in H_1', fontsize=12)
plt.ylabel('Density', fontsize=12)

# Make it look exactly like the EMNLP paper graph
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend(fontsize=12, frameon=False)
plt.tight_layout()

os.makedirs('results', exist_ok=True)
plt.savefig('results/dist_h1_sum_l9_h7.png', dpi=200, bbox_inches='tight')
print("Successfully generated TDA graph to results/dist_h1_sum_l9_h7.png")

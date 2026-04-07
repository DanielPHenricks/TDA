import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

# Load files
X_val = np.load('results/X_val.npy')
val_data = pd.read_csv('small_gpt_web/valid_5k.csv').head(500)
feats_per_head = 97

import features_colab
from features_colab import takens_layer_analysis, mapper_analysis
import matplotlib.pyplot as plt

cache_dir = "results/"

X_val_4d_base = X_val.reshape((len(val_data), 12, 12, feats_per_head))
X_val_4d = np.transpose(X_val_4d_base, (1, 2, 0, 3))
labels_used = val_data['label'].values

print("Running Takens...")
takens_layer_analysis(labels_used, X_val_4d, cache_dir, feature_idx=37)
print("Running Mapper...")
mapper_analysis(labels_used, X_val_4d, cache_dir)
print("Done.")

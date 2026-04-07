import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import os
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
X_train = np.load('results/X_train.npy')
X_val = np.load('results/X_val.npy')

train_data = pd.read_csv('small_gpt_web/test_5k.csv').head(500)
val_data = pd.read_csv('small_gpt_web/valid_5k.csv').head(500)

label_map = {lbl: i for i, lbl in enumerate(sorted(train_data['label'].unique()))}
y_train = train_data['label'].map(label_map).values
y_val = val_data['label'].map(label_map).values

print("Training Logistic Regression...")
scaler = StandardScaler()
X_train_s = np.nan_to_num(scaler.fit_transform(X_train))
X_val_s = np.nan_to_num(scaler.transform(X_val))

clf = LogisticRegressionCV(
    Cs=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 0.01, 0.05, 0.1, 0.5, 1.0],
    cv=5, max_iter=1000, n_jobs=-1, scoring='accuracy'
)
clf.fit(X_train_s, y_train)
val_preds = clf.predict(X_val_s)
val_acc = accuracy_score(y_val, val_preds)

print(f"Validation accuracy: {val_acc:.4f}\n")
print(classification_report(y_val, val_preds, target_names=[k for k, v in sorted(label_map.items(), key=lambda x: x[1])]))


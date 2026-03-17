#!/usr/bin/env python3
"""
Fine-tune BERT for binary classification (generated vs natural text),
then run persistent homology analysis on the fine-tuned model's attention.
"""

import os
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from ripser import ripser
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW

warnings.filterwarnings('ignore')

# =============================================================================
# Parameters
# =============================================================================

TRAIN_SAMPLES = 4000
TEST_SAMPLES = 500
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5
MODEL_NAME = "bert-base-uncased"
SAVE_DIR = "fine_tuned_bert"

subset = "test_5k"
input_dir = "small_gpt_web/"
output_dir = "small_gpt_web/"


# =============================================================================
# Dataset
# =============================================================================

def text_preprocessing(text):
    text = re.sub(r'(@.*?)[\s]', ' ', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = text_preprocessing(str(self.texts[idx]))
        encoding = self.tokenizer(
            text,
            return_tensors='pt',
            max_length=self.max_len,
            padding='max_length',
            truncation=True
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'token_type_ids': encoding['token_type_ids'].squeeze(),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }


# =============================================================================
# Persistence helpers
# =============================================================================

def attention_to_distance(matrix, ntokens):
    mat = matrix[:ntokens, :ntokens].copy()
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    mat /= row_sums
    mat = (mat > 0).astype(np.float32) * mat
    dist = 1.0 - mat
    np.fill_diagonal(dist, 0)
    dist = np.minimum(dist, dist.T)
    return dist


def extract_barcode_features(diagrams):
    features = {}
    for dim in range(min(2, len(diagrams))):
        dgm = diagrams[dim]
        finite_mask = np.isfinite(dgm[:, 1])
        dgm_finite = dgm[finite_mask]
        lengths = dgm_finite[:, 1] - dgm_finite[:, 0] if len(dgm_finite) > 0 else np.array([0.0])

        prefix = f"h{dim}"
        features[f"{prefix}_count"] = len(dgm_finite)
        features[f"{prefix}_sum"] = float(np.sum(lengths))
        features[f"{prefix}_mean"] = float(np.mean(lengths)) if len(lengths) > 0 else 0.0
        features[f"{prefix}_std"] = float(np.std(lengths)) if len(lengths) > 0 else 0.0
        features[f"{prefix}_max"] = float(np.max(lengths)) if len(lengths) > 0 else 0.0

        if len(lengths) > 0 and np.sum(lengths) > 0:
            probs = lengths / np.sum(lengths)
            probs = probs[probs > 0]
            features[f"{prefix}_entropy"] = float(-np.sum(probs * np.log(probs)))
        else:
            features[f"{prefix}_entropy"] = 0.0
    return features


# =============================================================================
# Main
# =============================================================================

def main():
    # --- Load data ---
    try:
        data = pd.read_csv(input_dir + subset + ".csv").reset_index(drop=True)
    except Exception:
        data = pd.read_csv(input_dir + subset + ".tsv", delimiter="\t", header=None)
        data.columns = ["0", "labels", "2", "sentence"]

    label_col = 'label' if 'label' in data.columns else 'labels'
    label_map = {lbl: i for i, lbl in enumerate(sorted(data[label_col].unique()))}
    label_names = {v: k for k, v in label_map.items()}
    data['label_id'] = data[label_col].map(label_map)
    print(f"Labels: {label_map}")

    # Split: first TRAIN_SAMPLES for training, next TEST_SAMPLES for analysis
    train_data = data.head(TRAIN_SAMPLES)
    test_data = data.iloc[TRAIN_SAMPLES:TRAIN_SAMPLES + TEST_SAMPLES].reset_index(drop=True)
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")
    print(f"Train label dist:\n{train_data[label_col].value_counts()}")
    print(f"Test label dist:\n{test_data[label_col].value_counts()}")

    # --- Device ---
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")

    # --- Tokenizer & Model ---
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME, do_lower_case=True)
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(label_map), output_attentions=True
    )
    model = model.to(device)

    # =========================================================================
    # PHASE 1: Fine-tune
    # =========================================================================
    print("\n" + "=" * 60)
    print("PHASE 1: Fine-tuning BERT")
    print("=" * 60)

    train_dataset = TextDataset(
        train_data['sentence'].values,
        train_data['label_id'].values,
        tokenizer, MAX_LEN
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=LR)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        total = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask,
                          token_type_ids=token_type_ids, labels=labels)

            loss = outputs.loss
            logits = outputs.logits

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)

        acc = correct / total
        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}, accuracy={acc:.4f}")

    # Save fine-tuned model
    os.makedirs(SAVE_DIR, exist_ok=True)
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    print(f"Model saved to {SAVE_DIR}/")

    # =========================================================================
    # PHASE 2: Persistence analysis on test set using fine-tuned model
    # =========================================================================
    print("\n" + "=" * 60)
    print("PHASE 2: Persistent homology on fine-tuned attention")
    print("=" * 60)

    model.eval()
    all_features = []

    for idx in tqdm(range(len(test_data)), desc="Computing persistence"):
        text = text_preprocessing(str(test_data['sentence'].iloc[idx]))

        inputs = tokenizer(
            [text], return_tensors='pt',
            max_length=MAX_LEN, padding='max_length', truncation=True
        )
        ntokens = (inputs['input_ids'][0] != tokenizer.pad_token_id).sum().item()
        ntokens = max(ntokens, 2)

        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        token_type_ids = inputs['token_type_ids'].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask,
                          token_type_ids=token_type_ids)

        attentions = [layer_att[0].cpu().numpy() for layer_att in outputs.attentions]

        sample_features = []
        for layer_att in attentions:
            layer_features = []
            for head_idx in range(layer_att.shape[0]):
                dist_mat = attention_to_distance(layer_att[head_idx], ntokens)
                result = ripser(dist_mat, maxdim=1, distance_matrix=True)
                feats = extract_barcode_features(result['dgms'])
                layer_features.append(feats)
            sample_features.append(layer_features)
        all_features.append(sample_features)

    # --- Organize ---
    labels = test_data[label_col].values
    n_layers = len(all_features[0])
    n_heads = len(all_features[0][0])
    feature_names = list(all_features[0][0][0].keys())

    feature_arrays = {}
    for fname in feature_names:
        arr = np.zeros((len(all_features), n_layers, n_heads))
        for s in range(len(all_features)):
            for l in range(n_layers):
                for h in range(n_heads):
                    arr[s, l, h] = all_features[s][l][h][fname]
        feature_arrays[fname] = arr

    os.makedirs(output_dir + 'features', exist_ok=True)
    np.save(output_dir + 'features/persistence_finetuned.npy', feature_arrays)

    # =========================================================================
    # PLOTS
    # =========================================================================
    unique_labels = sorted(set(labels))
    colors = {'natural': '#2196F3', 'generated': '#FF5722'}

    # Plot 1: Boxplots
    key_features = ['h0_count', 'h0_sum', 'h0_entropy', 'h1_count', 'h1_sum', 'h1_entropy']
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for i, fname in enumerate(key_features):
        ax = axes[i // 3][i % 3]
        arr = feature_arrays[fname]
        bp = ax.boxplot(
            [arr[labels == lbl].mean(axis=(1, 2)) for lbl in unique_labels],
            labels=unique_labels, patch_artist=True
        )
        for patch, lbl in zip(bp['boxes'], unique_labels):
            patch.set_facecolor(colors.get(lbl, '#999'))
            patch.set_alpha(0.6)
        ax.set_title(fname, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Fine-tuned BERT: Persistence Features (n={TEST_SAMPLES})', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir + 'persistence_finetuned_boxplots.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir}persistence_finetuned_boxplots.png")

    # Plot 2: Distributions
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8))
    for i, fname in enumerate(['h0_sum', 'h1_sum', 'h0_entropy', 'h1_entropy']):
        ax = axes2[i // 2][i % 2]
        arr = feature_arrays[fname].mean(axis=(1, 2))
        for lbl in unique_labels:
            ax.hist(arr[labels == lbl], bins=20, alpha=0.5, label=lbl,
                    color=colors.get(lbl, '#999'), density=True)
        ax.set_title(fname, fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig2.suptitle(f'Fine-tuned BERT: Feature Distributions (n={TEST_SAMPLES})', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir + 'persistence_finetuned_distributions.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir}persistence_finetuned_distributions.png")

    # --- Stats ---
    from scipy import stats
    print("\n" + "=" * 60)
    print("FEATURE COMPARISON (Fine-tuned BERT)")
    print("=" * 60)
    for fname in feature_names:
        arr = feature_arrays[fname].mean(axis=(1, 2))
        for lbl in unique_labels:
            vals = arr[labels == lbl]
            print(f"  {fname:15s} [{lbl:10s}]: mean={vals.mean():.4f} ± {vals.std():.4f}")
        mask_0, mask_1 = labels == unique_labels[0], labels == unique_labels[1]
        t_stat, p_val = stats.ttest_ind(arr[mask_0], arr[mask_1])
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        print(f"  {'':15s} p={p_val:.4f} {sig}\n")

    print("Done!")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Artificial Text Detection via Examining the Topology of Attention Maps
(EMNLP 2021) — Colab Version (Optimized)

Extracts three groups of TDA features from a frozen pre-trained BERT model's
attention maps and trains a Logistic Regression classifier.

Usage (Colab):
  1. Upload test_5k.csv and valid_5k.csv to /content/
  2. !pip install transformers ripser scikit-learn
  3. !python features_colab.py
"""

import os
import re
import warnings
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from ripser import ripser
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

warnings.filterwarnings('ignore')

# =============================================================================
# Parameters
# =============================================================================

MAX_LEN = 128
BATCH_SIZE = 64
MODEL_NAME = "bert-base-uncased"
THRESHOLDS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

input_dir = "./"
output_dir = "./"
TRAIN_FILE = "test_5k.csv"
VAL_FILE = "valid_5k.csv"
SAMPLE_SIZE = 500  # Set to None to process the full dataset

# Features per head:
#   Topological: 4 features × 9 thresholds = 36
#   Barcode:     8 features × 2 dims       = 16
#   Pattern:     5 patterns × 9 thresholds  = 45
# Total per head: 97
# Total: 12 layers × 12 heads × 97 = 13,968

# =============================================================================
# Text preprocessing
# =============================================================================

def text_preprocessing(text):
    text = re.sub(r'(@.*?)[\s]', ' ', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# =============================================================================
# Feature Group 1: Topological features (Section 4.1)
#   All computed with pure numpy/scipy — no NetworkX
# =============================================================================

def compute_topological_features(att_matrix, ntokens, thresholds):
    """
    For each threshold, compute from attention matrix:
      - n_edges (directed)
      - n_strongly_connected_components (directed)
      - β₀ (connected components of undirected graph)
      - β₁ = edges_undirected - vertices + β₀ (independent cycles)

    Uses scipy.sparse.csgraph instead of NetworkX for 100x speedup.
    """
    mat = att_matrix[:ntokens, :ntokens]
    n = ntokens
    features = np.empty(4 * len(thresholds), dtype=np.float32)

    for ti, t in enumerate(thresholds):
        # Directed adjacency (no self-loops)
        dir_adj = (mat >= t)
        np.fill_diagonal(dir_adj, False)

        # Directed edges
        n_edges_dir = int(np.sum(dir_adj))

        # SCC via scipy (directed)
        sparse_dir = csr_matrix(dir_adj.astype(np.float32))
        n_scc, _ = connected_components(sparse_dir, directed=True,
                                        connection='strong')

        # Undirected: edge if either direction
        undir_adj = np.logical_or(dir_adj, dir_adj.T)
        sparse_undir = csr_matrix(undir_adj.astype(np.float32))
        beta_0, _ = connected_components(sparse_undir, directed=False)

        # β₁ = |E_undirected| - |V| + β₀
        n_edges_undir = int(np.sum(undir_adj)) // 2  # symmetric, count once
        beta_1 = n_edges_undir - n + beta_0

        base = ti * 4
        features[base] = n_edges_dir
        features[base + 1] = n_scc
        features[base + 2] = beta_0
        features[base + 3] = beta_1

    return features


# =============================================================================
# Feature Group 2: Barcode features via Ripser (Section 4.2)
# =============================================================================

def attention_to_distance(att_matrix, ntokens):
    """
    Distance = 1 - max(W, W^T). Paper Appendix B: w ↦ 1 − w,
    symmetrized by taking max of either direction.
    """
    mat = att_matrix[:ntokens, :ntokens].astype(np.float64)
    mat_sym = np.maximum(mat, mat.T)
    dist = np.maximum(1.0 - mat_sym, 0.0)
    np.fill_diagonal(dist, 0.0)
    return dist


def extract_barcode_features(diagrams):
    """
    Per homology dimension (H0, H1):
      count, sum, mean, std, max, entropy, longest_birth, longest_death
    """
    features = np.zeros(16, dtype=np.float32)

    for dim in range(min(2, len(diagrams))):
        dgm = diagrams[dim]
        finite = dgm[np.isfinite(dgm[:, 1])]
        base = dim * 8

        if len(finite) > 0:
            lengths = finite[:, 1] - finite[:, 0]
            longest = np.argmax(lengths)
            total = float(np.sum(lengths))

            features[base] = len(finite)
            features[base + 1] = total
            features[base + 2] = float(np.mean(lengths))
            features[base + 3] = float(np.std(lengths))
            features[base + 4] = float(np.max(lengths))

            if total > 0:
                p = lengths / total
                p = p[p > 0]
                features[base + 5] = float(-np.sum(p * np.log(p)))

            features[base + 6] = float(finite[longest, 0])
            features[base + 7] = float(finite[longest, 1])

    return features


# =============================================================================
# Feature Group 3: Distance to attention patterns (Section 4.3)
# =============================================================================

def compute_pattern_features(att_matrix, ntokens, thresholds):
    """
    Normalized Frobenius distance from binarized attention to 5 canonical
    patterns at each threshold.
    """
    mat = att_matrix[:ntokens, :ntokens]
    n = ntokens

    # Pre-build pattern matrices
    prev_tok = np.zeros((n, n), dtype=np.float32)
    np.fill_diagonal(prev_tok[1:], 1.0)  # (i, i-1)

    next_tok = np.zeros((n, n), dtype=np.float32)
    np.fill_diagonal(next_tok[:, 1:], 1.0)  # (i, i+1)

    cls_tok = np.zeros((n, n), dtype=np.float32)
    cls_tok[:, 0] = 1.0
    np.fill_diagonal(cls_tok, 0.0)

    sep_tok = np.zeros((n, n), dtype=np.float32)
    sep_tok[:, n - 1] = 1.0
    np.fill_diagonal(sep_tok, 0.0)

    diag_tok = np.eye(n, dtype=np.float32)

    patterns = [prev_tok, next_tok, cls_tok, sep_tok, diag_tok]

    # Pre-compute pattern norms
    pattern_norms_sq = [float(np.sum(p * p)) for p in patterns]

    features = np.empty(5 * len(thresholds), dtype=np.float32)

    for ti, t in enumerate(thresholds):
        binary = (mat >= t).astype(np.float32)
        np.fill_diagonal(binary, 0.0)
        binary_norm_sq = float(np.sum(binary * binary))

        for pi, (pat, pnorm_sq) in enumerate(zip(patterns, pattern_norms_sq)):
            diff_sq = float(np.sum((binary - pat) ** 2))
            denom = binary_norm_sq + pnorm_sq
            features[ti * 5 + pi] = np.sqrt(diff_sq / denom) if denom > 0 else 0.0

    return features


# =============================================================================
# Combined per-head feature extraction
# =============================================================================

def compute_all_features_for_sample(attention, ntokens, thresholds):
    """
    For one sample: iterate over 12 layers × 12 heads, compute all 3
    feature groups, concatenate into a single flat vector.
    """
    n_layers, n_heads = attention.shape[0], attention.shape[1]
    n_thresh = len(thresholds)
    feats_per_head = 4 * n_thresh + 16 + 5 * n_thresh
    result = np.empty(n_layers * n_heads * feats_per_head, dtype=np.float32)

    pos = 0
    for layer in range(n_layers):
        for head in range(n_heads):
            att_mat = attention[layer, head]

            topo = compute_topological_features(att_mat, ntokens, thresholds)
            dist_mat = attention_to_distance(att_mat, ntokens)
            dgms = ripser(dist_mat, maxdim=1, distance_matrix=True)['dgms']
            barcode = extract_barcode_features(dgms)
            pattern = compute_pattern_features(att_mat, ntokens, thresholds)

            n = len(topo) + len(barcode) + len(pattern)
            result[pos:pos + n] = np.concatenate([topo, barcode, pattern])
            pos += n

    return result


# =============================================================================
# Main
# =============================================================================

def main():
    print("Loading datasets...")
    train_data = pd.read_csv(os.path.join(input_dir, TRAIN_FILE)).reset_index(drop=True)
    val_data = pd.read_csv(os.path.join(input_dir, VAL_FILE)).reset_index(drop=True)

    if SAMPLE_SIZE is not None:
        print(f"Limiting dataset to {SAMPLE_SIZE} samples...")
        train_data = train_data.head(SAMPLE_SIZE)
        val_data = val_data.head(SAMPLE_SIZE)

    label_map = {lbl: i for i, lbl in enumerate(sorted(train_data['label'].unique()))}
    train_data['label_id'] = train_data['label'].map(label_map)
    val_data['label_id'] = val_data['label'].map(label_map)

    print(f"Labels: {label_map}")
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device('cpu')
        print(f"Device: {device}")

    print(f"\nLoading frozen pre-trained {MODEL_NAME}...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME, do_lower_case=True)
    model = BertModel.from_pretrained(MODEL_NAME, output_attentions=True).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    cache_dir = os.path.join(output_dir, "tda_results/")
    os.makedirs(cache_dir, exist_ok=True)

    n_thresh = len(THRESHOLDS)
    feats_per_head = 4 * n_thresh + 16 + 5 * n_thresh
    total_feats = 12 * 12 * feats_per_head
    print(f"\nFeatures per head: {feats_per_head}")
    print(f"Total features per sample: {total_feats}")

    # =========================================================================
    # Process datasets on-the-fly (extract attention → compute features → discard)
    # =========================================================================
    from joblib import Parallel, delayed

    def process_dataset(sentences, desc):
        X = np.zeros((len(sentences), total_feats), dtype=np.float32)
        idx = 0
        for start in tqdm(range(0, len(sentences), BATCH_SIZE), desc=desc):
            batch_texts = [text_preprocessing(str(s))
                           for s in sentences[start:start + BATCH_SIZE]]
            inputs = tokenizer(batch_texts, return_tensors='pt',
                               max_length=MAX_LEN, padding='max_length',
                               truncation=True)

            with torch.no_grad():
                outputs = model(
                    inputs['input_ids'].to(device),
                    attention_mask=inputs['attention_mask'].to(device),
                    token_type_ids=inputs['token_type_ids'].to(device)
                )

            batch_ntokens = inputs['attention_mask'].sum(dim=1).numpy()

            # Move attention to CPU numpy immediately
            sample_data = []
            for i in range(len(batch_texts)):
                att = np.stack([layer[i].cpu().numpy()
                                for layer in outputs.attentions])
                nt = max(int(batch_ntokens[i]), 2)
                sample_data.append((att, nt))

            # Parallel feature computation across samples in batch
            batch_feats = Parallel(n_jobs=-1, prefer="threads")(
                delayed(compute_all_features_for_sample)(att, nt, THRESHOLDS)
                for att, nt in sample_data
            )

            for feats in batch_feats:
                X[idx] = feats
                idx += 1

        return X

    print("\n" + "=" * 60)
    print("Extracting attention & computing TDA features")
    print("=" * 60)

    X_train = process_dataset(train_data['sentence'].values, "Train")
    X_val = process_dataset(val_data['sentence'].values, "Val")

    # Save features
    np.save(os.path.join(cache_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(cache_dir, 'X_val.npy'), X_val)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    y_train = train_data['label_id'].values
    y_val = val_data['label_id'].values

    # =========================================================================
    # Train Logistic Regression
    # =========================================================================
    print("\n" + "=" * 60)
    print("Training Logistic Regression")
    print("=" * 60)

    scaler = StandardScaler()
    X_train_s = np.nan_to_num(scaler.fit_transform(X_train))
    X_val_s = np.nan_to_num(scaler.transform(X_val))

    clf = LogisticRegressionCV(
        Cs=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 0.01, 0.05, 0.1, 0.5, 1.0],
        cv=5, max_iter=1000, n_jobs=-1, scoring='accuracy'
    )
    clf.fit(X_train_s, y_train)

    train_acc = accuracy_score(y_train, clf.predict(X_train_s))
    val_preds = clf.predict(X_val_s)
    val_acc = accuracy_score(y_val, val_preds)

    label_names = [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]
    print(f"\nBest C: {clf.C_[0]:.6f}")
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Validation accuracy: {val_acc:.4f}\n")
    print(classification_report(y_val, val_preds, target_names=label_names))

    # =========================================================================
    # Feature group ablation
    # =========================================================================
    print("=" * 60)
    print("Feature Group Ablation")
    print("=" * 60)

    topo_size = 4 * n_thresh
    barcode_size = 16
    pattern_size = 5 * n_thresh

    def get_group_cols(group):
        cols = []
        for h in range(144):
            base = h * feats_per_head
            if group == 'topological':
                cols.extend(range(base, base + topo_size))
            elif group == 'barcode':
                cols.extend(range(base + topo_size, base + topo_size + barcode_size))
            elif group == 'pattern':
                cols.extend(range(base + topo_size + barcode_size, base + feats_per_head))
        return cols

    for name in ['topological', 'barcode', 'pattern']:
        cols = get_group_cols(name)
        c = LogisticRegressionCV(Cs=[1e-4, 1e-3, 0.01, 0.1, 1.0],
                                  cv=5, max_iter=1000, n_jobs=-1)
        c.fit(X_train_s[:, cols], y_train)
        acc = accuracy_score(y_val, c.predict(X_val_s[:, cols]))
        print(f"  {name:15s}: {acc:.4f}  ({len(cols)} features)")

    # =========================================================================
    # Per-head accuracy heatmap
    # =========================================================================
    print("\nComputing per-head accuracies...")
    head_accs = np.zeros((12, 12))
    for layer in tqdm(range(12), desc="Head-wise eval"):
        for head in range(12):
            h = layer * 12 + head
            cols = list(range(h * feats_per_head, (h + 1) * feats_per_head))
            lr = LogisticRegression(C=0.01, max_iter=500)
            lr.fit(X_train_s[:, cols], y_train)
            head_accs[layer, head] = accuracy_score(y_val, lr.predict(X_val_s[:, cols]))

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(head_accs, cmap='RdYlGn', vmin=0.45, vmax=0.75, aspect='auto')
    ax.set_xlabel('Attention Head', fontsize=12)
    ax.set_ylabel('Layer', fontsize=12)
    ax.set_title('Per-Head Classification Accuracy (LR)', fontsize=14, fontweight='bold')
    ax.set_xticks(range(12))
    ax.set_yticks(range(12))
    plt.colorbar(im, ax=ax, label='Val Accuracy')
    for i in range(12):
        for j in range(12):
            ax.text(j, i, f'{head_accs[i, j]:.2f}', ha='center', va='center',
                    fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(cache_dir, 'head_accuracy_heatmap.png'), dpi=150)
    print(f"Saved heatmap.")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    best_layer, best_head = np.unravel_index(np.argmax(head_accs), (12, 12))
    print(f"Model:          {MODEL_NAME} (frozen)")
    print(f"Train/Val:      {len(train_data)} / {len(val_data)}")
    print(f"Total features: {total_feats}")
    print(f"Val accuracy:   {val_acc:.4f}")
    print(f"Best head:      Layer {best_layer}, Head {best_head} "
          f"({head_accs[best_layer, best_head]:.4f})")

    # Zip results
    zip_path = "tda_features_results.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in os.listdir(cache_dir):
            zf.write(os.path.join(cache_dir, f), f)

    try:
        from google.colab import files
        files.download(zip_path)
    except ImportError:
        print(f"Results saved to {zip_path}")

    print("\nDone!")


if __name__ == '__main__':
    main()

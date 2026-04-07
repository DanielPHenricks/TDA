#!/usr/bin/env python3
"""
Artificial Text Detection via Examining the Topology of Attention Maps
(EMNLP 2021) — Reimplementation

Extracts three groups of TDA features from a frozen pre-trained BERT model's
attention maps and trains a Logistic Regression classifier to distinguish
generated from natural text.

Feature groups:
  1. Topological features at multiple thresholds (Section 4.1)
  2. Barcode features from persistent homology via Ripser (Section 4.2)
  3. Distance-to-attention-pattern features (Section 4.3)

Usage:
    python3 features_calculation_by_thresholds.py
"""

import os
import re
import warnings

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
from ripser import ripser
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

warnings.filterwarnings('ignore')

# =============================================================================
# Parameters
# =============================================================================

MAX_LEN = 128
BATCH_SIZE = 32
MODEL_NAME = "bert-base-uncased"

# Thresholds for topological features (Section 4.1)
THRESHOLDS = np.arange(0.0, 1.0, 0.1)

input_dir = "small_gpt_web/"
output_dir = "small_gpt_web/"

TRAIN_FILE = "test_5k.csv"
VAL_FILE = "valid_5k.csv"


# =============================================================================
# Text preprocessing
# =============================================================================

def text_preprocessing(text):
    text = re.sub(r'(@.*?)[\s]', ' ', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# =============================================================================
# Attention extraction
# =============================================================================

def extract_attention_batched(model, tokenizer, sentences, device, max_len=128,
                              batch_size=32):
    """
    Extract attention weights from frozen pre-trained BERT for a list of
    sentences. Returns a list of attention arrays and token counts.

    Returns:
        all_attentions: list of np.array [n_layers, n_heads, ntokens, ntokens]
        all_ntokens:    list of int
    """
    model.eval()
    all_attentions = []
    all_ntokens = []

    for start in tqdm(range(0, len(sentences), batch_size),
                      desc="Extracting attention"):
        batch_texts = [text_preprocessing(str(s))
                       for s in sentences[start:start + batch_size]]

        inputs = tokenizer(
            batch_texts,
            return_tensors='pt',
            max_length=max_len,
            padding='max_length',
            truncation=True
        )

        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        token_type_ids = inputs['token_type_ids'].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)

        # outputs.attentions: tuple of (batch, heads, seq, seq) per layer
        attentions = outputs.attentions

        batch_ntokens = attention_mask.sum(dim=1).cpu().numpy()

        for i in range(len(batch_texts)):
            # Stack layers: [n_layers, n_heads, seq_len, seq_len]
            sample_att = np.stack(
                [layer_att[i].cpu().numpy() for layer_att in attentions]
            )
            nt = max(int(batch_ntokens[i]), 2)
            all_attentions.append(sample_att)
            all_ntokens.append(nt)

    return all_attentions, all_ntokens


# =============================================================================
# Feature Group 1: Topological features at thresholds (Section 4.1)
# =============================================================================

def compute_topological_features(att_matrix, ntokens, thresholds):
    """
    For a single attention head matrix, compute topological features at each
    threshold. Returns a flat feature vector.

    At each threshold t, from the DIRECTED graph:
      - number of edges
      - number of strongly connected components
      - number of simple cycles (capped at 500)

    At each threshold t, from the UNDIRECTED graph:
      - β₀ (number of connected components)
      - β₁ (number of independent cycles = |E| - |V| + β₀)
    """
    mat = att_matrix[:ntokens, :ntokens].copy()
    features = []

    for t in thresholds:
        # Directed graph: edge j→i exists if w_ij >= t
        dir_adj = (mat >= t).astype(np.int8)
        np.fill_diagonal(dir_adj, 0)
        G_dir = nx.from_numpy_array(dir_adj, create_using=nx.DiGraph())

        n_edges_dir = G_dir.number_of_edges()
        n_scc = nx.number_strongly_connected_components(G_dir)

        # Cap cycle counting
        n_cycles = 0
        for _ in nx.simple_cycles(G_dir):
            n_cycles += 1
            if n_cycles >= 500:
                break

        # Undirected graph: edge exists if directed edge in either direction
        undir_adj = np.maximum(dir_adj, dir_adj.T)
        G_undir = nx.from_numpy_array(undir_adj)

        beta_0 = nx.number_connected_components(G_undir)
        n_edges_undir = G_undir.number_of_edges()
        n_vertices = G_undir.number_of_nodes()
        beta_1 = n_edges_undir - n_vertices + beta_0

        features.extend([n_edges_dir, n_scc, n_cycles, beta_0, beta_1])

    return np.array(features, dtype=np.float32)


# =============================================================================
# Feature Group 2: Barcode features via Ripser (Section 4.2)
# =============================================================================

def attention_to_distance(att_matrix, ntokens):
    """
    Convert attention matrix to distance matrix for Ripser.

    Paper (Appendix B): "the increasing filtration is obtained by
    reversing the attention matrix weights: w ↦ 1 − w"

    Undirected: edge weight = max of either direction.
    Distance = 1 - weight (high attention → low distance → early in filtration).
    """
    mat = att_matrix[:ntokens, :ntokens].copy().astype(np.float64)
    # Symmetrize by taking max (edge exists if either direction has weight)
    mat_sym = np.maximum(mat, mat.T)
    # Convert to distance
    dist = 1.0 - mat_sym
    np.fill_diagonal(dist, 0.0)
    # Ensure no negative distances from floating point
    dist = np.maximum(dist, 0.0)
    return dist


def extract_barcode_features(diagrams):
    """
    Extract barcode features for H0 and H1 persistent homology groups.

    Per homology dimension:
      - count: number of finite bars
      - sum: sum of bar lengths
      - mean: mean bar length
      - std: std of bar lengths (variance proxy)
      - max: maximum bar length
      - entropy: barcode entropy
      - longest_birth: birth time of longest bar
      - longest_death: death time of longest bar
    """
    features = []

    for dim in range(min(2, len(diagrams))):
        dgm = diagrams[dim]
        finite_mask = np.isfinite(dgm[:, 1])
        dgm_finite = dgm[finite_mask]

        if len(dgm_finite) > 0:
            lengths = dgm_finite[:, 1] - dgm_finite[:, 0]
            longest_idx = np.argmax(lengths)

            count = len(dgm_finite)
            bar_sum = float(np.sum(lengths))
            bar_mean = float(np.mean(lengths))
            bar_std = float(np.std(lengths))
            bar_max = float(np.max(lengths))
            longest_birth = float(dgm_finite[longest_idx, 0])
            longest_death = float(dgm_finite[longest_idx, 1])

            if bar_sum > 0:
                probs = lengths / bar_sum
                probs = probs[probs > 0]
                entropy = float(-np.sum(probs * np.log(probs)))
            else:
                entropy = 0.0
        else:
            count = 0
            bar_sum = bar_mean = bar_std = bar_max = 0.0
            longest_birth = longest_death = 0.0
            entropy = 0.0

        features.extend([count, bar_sum, bar_mean, bar_std, bar_max,
                         entropy, longest_birth, longest_death])

    # If H1 was not computed (shouldn't happen with maxdim=1), pad zeros
    while len(features) < 16:
        features.extend([0.0] * 8)

    return np.array(features, dtype=np.float32)


# =============================================================================
# Feature Group 3: Distance to attention patterns (Section 4.3)
# =============================================================================

def build_pattern_matrices(ntokens):
    """
    Build incidence matrices for canonical attention patterns (Clark et al.).

    Patterns:
      - previous_token: edge (i+1) → i
      - next_token:     edge i → (i+1)
      - cls_token:      every token attends to [CLS] (vertex 0)
      - sep_token:      every token attends to [SEP] (last real token)
      - diagonal:       self-attention (identity)
    """
    patterns = {}
    n = ntokens

    # Previous token
    prev_tok = np.zeros((n, n), dtype=np.float32)
    for i in range(1, n):
        prev_tok[i, i - 1] = 1.0
    patterns['prev_token'] = prev_tok

    # Next token
    next_tok = np.zeros((n, n), dtype=np.float32)
    for i in range(n - 1):
        next_tok[i, i + 1] = 1.0
    patterns['next_token'] = next_tok

    # CLS token (all attend to position 0)
    cls_tok = np.zeros((n, n), dtype=np.float32)
    cls_tok[:, 0] = 1.0
    np.fill_diagonal(cls_tok, 0.0)
    patterns['cls_token'] = cls_tok

    # SEP token (all attend to last token)
    sep_tok = np.zeros((n, n), dtype=np.float32)
    sep_tok[:, n - 1] = 1.0
    np.fill_diagonal(sep_tok, 0.0)
    patterns['sep_token'] = sep_tok

    # Diagonal (self-attention)
    diag_tok = np.eye(n, dtype=np.float32)
    patterns['diagonal'] = diag_tok

    return patterns


def frobenius_distance(A, B):
    """
    Normalized Frobenius distance between binary/real matrices.
    d(A, B) = ||A - B||_F / sqrt(||A||_F^2 + ||B||_F^2)
    """
    diff_norm_sq = np.sum((A - B) ** 2)
    sum_norm_sq = np.sum(A ** 2) + np.sum(B ** 2)
    if sum_norm_sq == 0:
        return 0.0
    return float(np.sqrt(diff_norm_sq / sum_norm_sq))


def compute_pattern_features(att_matrix, ntokens, thresholds):
    """
    Compute distance-to-pattern features for each threshold.
    At each threshold, binarize the attention graph and compute the
    normalized Frobenius distance to each canonical pattern.
    """
    mat = att_matrix[:ntokens, :ntokens].copy()
    patterns = build_pattern_matrices(ntokens)
    features = []

    for t in thresholds:
        binary_graph = (mat >= t).astype(np.float32)
        np.fill_diagonal(binary_graph, 0.0)

        for pattern_name in ['prev_token', 'next_token', 'cls_token',
                             'sep_token', 'diagonal']:
            dist = frobenius_distance(binary_graph, patterns[pattern_name])
            features.append(dist)

    return np.array(features, dtype=np.float32)


# =============================================================================
# Full feature extraction pipeline
# =============================================================================

def compute_all_features_for_sample(attention, ntokens, thresholds):
    """
    Compute all three feature groups for a single sample.
    attention: [n_layers, n_heads, seq_len, seq_len]

    Returns a flat feature vector: concatenation of all features across
    all layers and heads.
    """
    n_layers = attention.shape[0]
    n_heads = attention.shape[1]

    all_features = []

    for layer in range(n_layers):
        for head in range(n_heads):
            att_mat = attention[layer, head]

            # Group 1: Topological features
            topo_feats = compute_topological_features(att_mat, ntokens,
                                                      thresholds)

            # Group 2: Barcode features
            dist_mat = attention_to_distance(att_mat, ntokens)
            result = ripser(dist_mat, maxdim=1, distance_matrix=True)
            barcode_feats = extract_barcode_features(result['dgms'])

            # Group 3: Pattern distance features
            pattern_feats = compute_pattern_features(att_mat, ntokens,
                                                     thresholds)

            all_features.append(np.concatenate([
                topo_feats, barcode_feats, pattern_feats
            ]))

    return np.concatenate(all_features)


# =============================================================================
# Main
# =============================================================================

def main():
    # --- Load data ---
    print("Loading datasets...")
    train_data = pd.read_csv(input_dir + TRAIN_FILE).reset_index(drop=True)
    val_data = pd.read_csv(input_dir + VAL_FILE).reset_index(drop=True)

    label_col = 'label'
    label_map = {lbl: i for i, lbl in enumerate(sorted(train_data[label_col].unique()))}
    train_data['label_id'] = train_data[label_col].map(label_map)
    val_data['label_id'] = val_data[label_col].map(label_map)

    print(f"Labels: {label_map}")
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    print(f"Train dist:\n{train_data[label_col].value_counts().to_string()}")
    print(f"Val dist:\n{val_data[label_col].value_counts().to_string()}")

    # --- Device ---
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")

    # --- Load frozen pre-trained BERT ---
    print(f"\nLoading frozen pre-trained {MODEL_NAME}...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME, do_lower_case=True)
    model = BertModel.from_pretrained(MODEL_NAME, output_attentions=True)
    model = model.to(device)
    model.eval()
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    print("Model loaded (frozen, no fine-tuning).")

    # =========================================================================
    # PHASE 1: Extract attention weights
    # =========================================================================
    print("\n" + "=" * 60)
    print("PHASE 1: Extracting attention from pre-trained BERT")
    print("=" * 60)

    cache_dir = output_dir + "attention_cache/"
    os.makedirs(cache_dir, exist_ok=True)

    train_cache = cache_dir + "train_attentions.npz"
    val_cache = cache_dir + "val_attentions.npz"

    if os.path.exists(train_cache):
        print(f"Loading cached train attentions from {train_cache}...")
        loaded = np.load(train_cache, allow_pickle=True)
        train_attentions = list(loaded['attentions'])
        train_ntokens = list(loaded['ntokens'])
    else:
        train_attentions, train_ntokens = extract_attention_batched(
            model, tokenizer, train_data['sentence'].values, device,
            max_len=MAX_LEN, batch_size=BATCH_SIZE
        )
        np.savez_compressed(train_cache,
                            attentions=np.array(train_attentions, dtype=object),
                            ntokens=np.array(train_ntokens))
        print(f"Cached train attentions to {train_cache}")

    if os.path.exists(val_cache):
        print(f"Loading cached val attentions from {val_cache}...")
        loaded = np.load(val_cache, allow_pickle=True)
        val_attentions = list(loaded['attentions'])
        val_ntokens = list(loaded['ntokens'])
    else:
        val_attentions, val_ntokens = extract_attention_batched(
            model, tokenizer, val_data['sentence'].values, device,
            max_len=MAX_LEN, batch_size=BATCH_SIZE
        )
        np.savez_compressed(val_cache,
                            attentions=np.array(val_attentions, dtype=object),
                            ntokens=np.array(val_ntokens))
        print(f"Cached val attentions to {val_cache}")

    # Free GPU memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # =========================================================================
    # PHASE 2: Compute TDA features
    # =========================================================================
    print("\n" + "=" * 60)
    print("PHASE 2: Computing TDA features (all 3 groups)")
    print("=" * 60)

    n_thresholds = len(THRESHOLDS)
    # Feature dimensions per head:
    #   Topo: 5 features × n_thresholds
    #   Barcode: 16 features (8 per homology dim × 2 dims)
    #   Pattern: 5 patterns × n_thresholds
    feats_per_head = 5 * n_thresholds + 16 + 5 * n_thresholds
    total_feats = 12 * 12 * feats_per_head
    print(f"Features per head: {feats_per_head}")
    print(f"Total features per sample: {total_feats} "
          f"(12 layers × 12 heads × {feats_per_head})")

    train_feat_cache = cache_dir + "train_features.npy"
    val_feat_cache = cache_dir + "val_features.npy"

    if os.path.exists(train_feat_cache):
        print(f"Loading cached train features from {train_feat_cache}...")
        X_train = np.load(train_feat_cache)
    else:
        X_train = np.zeros((len(train_attentions), total_feats),
                           dtype=np.float32)
        for i in tqdm(range(len(train_attentions)),
                      desc="Train features"):
            X_train[i] = compute_all_features_for_sample(
                train_attentions[i], train_ntokens[i], THRESHOLDS
            )
        np.save(train_feat_cache, X_train)
        print(f"Cached train features to {train_feat_cache}")

    if os.path.exists(val_feat_cache):
        print(f"Loading cached val features from {val_feat_cache}...")
        X_val = np.load(val_feat_cache)
    else:
        X_val = np.zeros((len(val_attentions), total_feats),
                         dtype=np.float32)
        for i in tqdm(range(len(val_attentions)),
                      desc="Val features"):
            X_val[i] = compute_all_features_for_sample(
                val_attentions[i], val_ntokens[i], THRESHOLDS
            )
        np.save(val_feat_cache, X_val)
        print(f"Cached val features to {val_feat_cache}")

    y_train = train_data['label_id'].values
    y_val = val_data['label_id'].values

    # =========================================================================
    # PHASE 3: Train Logistic Regression
    # =========================================================================
    print("\n" + "=" * 60)
    print("PHASE 3: Training Logistic Regression classifier")
    print("=" * 60)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Replace NaN/inf from standardization
    X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0,
                                   neginf=0.0)
    X_val_scaled = np.nan_to_num(X_val_scaled, nan=0.0, posinf=0.0,
                                 neginf=0.0)

    print("Training LR with cross-validated regularization...")
    clf = LogisticRegressionCV(
        Cs=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 0.01, 0.05, 0.1, 0.5, 1.0],
        cv=5,
        max_iter=1000,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )
    clf.fit(X_train_scaled, y_train)

    train_acc = accuracy_score(y_train, clf.predict(X_train_scaled))
    val_preds = clf.predict(X_val_scaled)
    val_acc = accuracy_score(y_val, val_preds)

    print(f"\nBest C: {clf.C_[0]:.6f}")
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Validation accuracy: {val_acc:.4f}")

    label_names = [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]
    print(f"\nClassification Report:\n")
    print(classification_report(y_val, val_preds, target_names=label_names))

    # =========================================================================
    # PHASE 4: Feature group ablation
    # =========================================================================
    print("\n" + "=" * 60)
    print("PHASE 4: Feature group ablation")
    print("=" * 60)

    topo_size = 5 * n_thresholds
    barcode_size = 16
    pattern_size = 5 * n_thresholds

    def get_feature_group_indices(group):
        """Get column indices for a specific feature group across all heads."""
        indices = []
        for head_idx in range(144):  # 12 layers × 12 heads
            base = head_idx * feats_per_head
            if group == 'topological':
                indices.extend(range(base, base + topo_size))
            elif group == 'barcode':
                indices.extend(range(base + topo_size,
                                     base + topo_size + barcode_size))
            elif group == 'pattern':
                indices.extend(range(base + topo_size + barcode_size,
                                     base + feats_per_head))
        return indices

    for group_name in ['topological', 'barcode', 'pattern']:
        idx = get_feature_group_indices(group_name)
        X_tr_g = X_train_scaled[:, idx]
        X_va_g = X_val_scaled[:, idx]

        clf_g = LogisticRegressionCV(
            Cs=[1e-4, 1e-3, 0.01, 0.1, 1.0],
            cv=5, max_iter=1000, n_jobs=-1, verbose=0
        )
        clf_g.fit(X_tr_g, y_train)
        acc_g = accuracy_score(y_val, clf_g.predict(X_va_g))
        print(f"  {group_name:15s} features only: val_acc = {acc_g:.4f} "
              f"({len(idx)} features)")

    # =========================================================================
    # PHASE 5: Diagnostic plots
    # =========================================================================
    print("\n" + "=" * 60)
    print("PHASE 5: Generating diagnostic plots")
    print("=" * 60)

    os.makedirs(output_dir + 'features', exist_ok=True)

    # --- Plot 1: Per-head accuracy heatmap ---
    # For each layer×head, train a simple LR on just that head's features
    head_accs = np.zeros((12, 12))
    for layer in tqdm(range(12), desc="Head-wise evaluation"):
        for head in range(12):
            head_idx = layer * 12 + head
            base = head_idx * feats_per_head
            cols = list(range(base, base + feats_per_head))
            X_tr_h = X_train_scaled[:, cols]
            X_va_h = X_val_scaled[:, cols]
            from sklearn.linear_model import LogisticRegression
            clf_h = LogisticRegression(C=0.01, max_iter=500)
            clf_h.fit(X_tr_h, y_train)
            head_accs[layer, head] = accuracy_score(y_val, clf_h.predict(X_va_h))

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(head_accs, cmap='RdYlGn', vmin=0.45, vmax=0.75,
                   aspect='auto')
    ax.set_xlabel('Attention Head', fontsize=12)
    ax.set_ylabel('Layer', fontsize=12)
    ax.set_title('Per-Head Classification Accuracy', fontsize=14,
                 fontweight='bold')
    ax.set_xticks(range(12))
    ax.set_yticks(range(12))
    plt.colorbar(im, ax=ax, label='Validation Accuracy')

    # Annotate
    for i in range(12):
        for j in range(12):
            ax.text(j, i, f'{head_accs[i, j]:.2f}', ha='center', va='center',
                    fontsize=7, color='black')

    plt.tight_layout()
    plt.savefig(output_dir + 'head_accuracy_heatmap.png', dpi=150,
                bbox_inches='tight')
    print(f"Saved: {output_dir}head_accuracy_heatmap.png")

    # --- Plot 2: Top discriminative heads — barcode sum distributions ---
    # Find top 6 heads by accuracy
    flat_accs = head_accs.flatten()
    top_heads = np.argsort(flat_accs)[-6:][::-1]

    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 8))
    colors = {'generated': '#FF5722', 'natural': '#2196F3'}

    for plot_idx, head_flat in enumerate(top_heads):
        layer = head_flat // 12
        head = head_flat % 12
        ax = axes2[plot_idx // 3][plot_idx % 3]

        # Get H0 sum (index 1 in barcode features, after topo features)
        base = head_flat * feats_per_head + topo_size
        h0_sum_col = base + 1  # count=0, sum=1

        for lbl_name, lbl_id in label_map.items():
            mask = y_val == lbl_id
            vals = X_val[mask, h0_sum_col]
            ax.hist(vals, bins=25, alpha=0.5, label=lbl_name,
                    color=colors.get(lbl_name, '#999'), density=True)

        ax.set_title(f'Layer {layer}, Head {head}\n'
                     f'(acc={head_accs[layer, head]:.3f})',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('H0 bar sum')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig2.suptitle('H0 Bar Sum Distribution — Top 6 Discriminative Heads',
                  fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir + 'top_heads_h0_sum.png', dpi=150,
                bbox_inches='tight')
    print(f"Saved: {output_dir}top_heads_h0_sum.png")

    # --- Plot 3: Feature group comparison bar chart ---
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    group_names_plot = ['Topological', 'Barcode', 'Pattern', 'All Features']
    group_accs = []
    for group_name in ['topological', 'barcode', 'pattern']:
        idx = get_feature_group_indices(group_name)
        X_tr_g = X_train_scaled[:, idx]
        X_va_g = X_val_scaled[:, idx]
        clf_g = LogisticRegressionCV(Cs=[1e-4, 1e-3, 0.01, 0.1, 1.0],
                                     cv=5, max_iter=1000, n_jobs=-1)
        clf_g.fit(X_tr_g, y_train)
        group_accs.append(accuracy_score(y_val, clf_g.predict(X_va_g)))
    group_accs.append(val_acc)

    bar_colors = ['#FF9800', '#4CAF50', '#2196F3', '#9C27B0']
    bars = ax3.bar(group_names_plot, group_accs, color=bar_colors, alpha=0.8,
                   edgecolor='white', linewidth=2)
    for bar, acc in zip(bars, group_accs):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    ax3.set_ylabel('Validation Accuracy', fontsize=12)
    ax3.set_title('Feature Group Comparison', fontsize=14, fontweight='bold')
    ax3.set_ylim(0.45, max(group_accs) + 0.05)
    ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5,
                label='Random baseline')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir + 'feature_group_comparison.png', dpi=150,
                bbox_inches='tight')
    print(f"Saved: {output_dir}feature_group_comparison.png")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Model:             {MODEL_NAME} (frozen, pre-trained)")
    print(f"Train samples:     {len(train_data)}")
    print(f"Val samples:       {len(val_data)}")
    print(f"Total features:    {total_feats}")
    print(f"Best LR C:         {clf.C_[0]:.6f}")
    print(f"Train accuracy:    {train_acc:.4f}")
    print(f"Val accuracy:      {val_acc:.4f}")
    print(f"Best single head:  Layer {np.unravel_index(np.argmax(head_accs), (12,12))[0]}, "
          f"Head {np.unravel_index(np.argmax(head_accs), (12,12))[1]} "
          f"({np.max(head_accs):.4f})")
    print("Done!")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Compute and plot H_0 (Betti-0) features from pre-computed attention matrices.

H_0 = number of connected components of the undirected thresholded attention graph.
"""

import os
import re
import warnings

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings('ignore')


# =============================================================================
# Parameters
# =============================================================================

max_tokens_amount = 128
MAX_SAMPLES = 100  # Only process this many samples
layers_of_interest = list(range(12))
thresholds_array = [0.025, 0.05, 0.1, 0.25, 0.5, 0.75]
model_name = "bert-base-uncased"

subset = "test_5k"
input_dir = "small_gpt_web/"
output_dir = "small_gpt_web/"
batch_size = 10
DUMP_SIZE = 100

r_file = (output_dir + 'attentions/' + subset + "_all_heads_" +
          str(len(layers_of_interest)) + "_layers_MAX_LEN_" +
          str(max_tokens_amount) + "_" + model_name)


# =============================================================================
# Helper functions
# =============================================================================

def cutoff_matrix(matrix, ntokens):
    """Return normalized submatrix of first ntokens."""
    matrix = matrix[:ntokens, :ntokens]
    matrix /= matrix.sum(axis=1, keepdims=True)
    return matrix


def compute_h0_single(adj_matrix, thresholds, ntokens):
    """Compute H_0 (number of connected components) at each threshold for one attention head."""
    h0_values = []
    mat = cutoff_matrix(adj_matrix.copy(), ntokens)
    for thr in thresholds:
        binary = (mat >= thr).astype(np.int8)
        g = nx.from_numpy_array(np.array(binary))
        h0_values.append(nx.number_connected_components(g))
    return h0_values


# =============================================================================
# Main
# =============================================================================

def main():
    # Load data for token lengths
    try:
        data = pd.read_csv(input_dir + subset + ".csv").reset_index(drop=True)
    except Exception:
        data = pd.read_csv(input_dir + subset + ".tsv", delimiter="\t", header=None)
        data.columns = ["0", "labels", "2", "sentence"]

    print(f"Loaded {len(data)} samples")

    # Estimate token lengths (capped at max_tokens_amount)
    ntokens_array = data['sentence'].apply(
        lambda x: min(len(x.split()), max_tokens_amount)
    ).values

    # Find attention files
    adj_filenames = sorted([
        output_dir + 'attentions/' + f
        for f in os.listdir(output_dir + 'attentions/')
        if r_file in (output_dir + 'attentions/' + f)
    ], key=lambda x: int(x.split('_')[-1].split('of')[0][4:].strip()))

    print(f"Found {len(adj_filenames)} attention files")
    assert len(adj_filenames) > 0, f"No attention files found matching: {r_file}"

    # Compute H_0 for all samples
    # Shape: [num_samples, num_layers, num_heads, num_thresholds]
    all_h0 = []
    sample_idx = 0

    for file_i, filename in enumerate(adj_filenames):
        print(f"\nProcessing {filename}...")
        attention_data = np.load(filename, allow_pickle=True)
        # Shape: (num_samples_in_file, num_layers, num_heads, max_tokens, max_tokens)
        n_samples = attention_data.shape[0]
        n_layers = attention_data.shape[1]
        n_heads = attention_data.shape[2]

        for s in tqdm(range(n_samples), desc=f"File {file_i+1}/{len(adj_filenames)}"):
            if sample_idx >= MAX_SAMPLES:
                break
            ntok = ntokens_array[sample_idx] if sample_idx < len(ntokens_array) else max_tokens_amount
            sample_h0 = []
            for layer in range(n_layers):
                layer_h0 = []
                for head in range(n_heads):
                    h0_vals = compute_h0_single(
                        attention_data[s, layer, head],
                        thresholds_array,
                        ntok
                    )
                    layer_h0.append(h0_vals)
                sample_h0.append(layer_h0)
            all_h0.append(sample_h0)
            sample_idx += 1

        if sample_idx >= MAX_SAMPLES:
            break

    all_h0 = np.array(all_h0)  # (samples, layers, heads, thresholds)
    print(f"\nH_0 array shape: {all_h0.shape}")

    # Save H_0 features
    h0_file = output_dir + 'features/h0_features.npy'
    os.makedirs(os.path.dirname(h0_file), exist_ok=True)
    np.save(h0_file, all_h0)
    print(f"Saved H_0 features to {h0_file}")

    # ==========================================================================
    # Plot H_0 features
    # ==========================================================================

    # If data has labels, split by label for comparison
    has_labels = 'label' in data.columns or 'labels' in data.columns
    label_col = 'label' if 'label' in data.columns else 'labels'

    # Plot 1: Mean H_0 across all samples, averaged over heads, for each layer
    fig, axes = plt.subplots(3, 4, figsize=(16, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    for layer in range(min(12, all_h0.shape[1])):
        ax = axes[layer]
        # Average over all heads: (samples, thresholds)
        h0_layer = all_h0[:, layer, :, :].mean(axis=1)

        if has_labels:
            labels = data[label_col].values[:len(all_h0)]
            unique_labels = sorted(set(labels))
            for lbl in unique_labels:
                mask = labels == lbl
                mean_h0 = h0_layer[mask].mean(axis=0)
                ax.plot(thresholds_array, mean_h0, marker='o', label=str(lbl), markersize=4)
            ax.legend(fontsize=7)
        else:
            mean_h0 = h0_layer.mean(axis=0)
            ax.plot(thresholds_array, mean_h0, marker='o', markersize=4)

        ax.set_title(f'Layer {layer}', fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Mean H₀ (Connected Components) by Threshold per Layer', fontsize=14)
    fig.supxlabel('Threshold')
    fig.supylabel('H₀ (# Connected Components)')
    plt.tight_layout()
    plt.savefig(output_dir + 'h0_by_layer.png', dpi=150, bbox_inches='tight')
    print(f"Saved plot: {output_dir}h0_by_layer.png")
    plt.show()

    # Plot 2: Mean H_0 averaged over all layers and heads
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    h0_global = all_h0.mean(axis=(1, 2))  # (samples, thresholds)

    if has_labels:
        labels = data[label_col].values[:len(all_h0)]
        unique_labels = sorted(set(labels))
        for lbl in unique_labels:
            mask = labels == lbl
            mean_h0 = h0_global[mask].mean(axis=0)
            std_h0 = h0_global[mask].std(axis=0)
            ax2.plot(thresholds_array, mean_h0, marker='o', label=str(lbl), linewidth=2)
            ax2.fill_between(thresholds_array, mean_h0 - std_h0, mean_h0 + std_h0, alpha=0.2)
        ax2.legend()
    else:
        mean_h0 = h0_global.mean(axis=0)
        std_h0 = h0_global.std(axis=0)
        ax2.plot(thresholds_array, mean_h0, marker='o', linewidth=2)
        ax2.fill_between(thresholds_array, mean_h0 - std_h0, mean_h0 + std_h0, alpha=0.2)

    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('H₀ (# Connected Components)')
    ax2.set_title('Mean H₀ Across All Layers and Heads')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir + 'h0_global.png', dpi=150, bbox_inches='tight')
    print(f"Saved plot: {output_dir}h0_global.png")
    plt.show()

    print("Done!")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Fine-tune BERT for binary classification (generated vs natural text).
Designed to run on Google Colab with GPU. Trains on the full dataset
and exports the fine-tuned model as a downloadable zip.

Usage (Colab):
  1. Upload test_5k.csv and valid_5k.csv to Colab (or mount Google Drive)
  2. Run this script
  3. Download fine_tuned_bert.zip when complete
"""

# --- Install dependencies (uncomment on Colab) ---
# !pip install transformers torch --quiet

import os
import re
import zipfile

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification

# =============================================================================
# Parameters
# =============================================================================

MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 10
LR = 2e-5
WEIGHT_DECAY = 0.01
MODEL_NAME = "bert-base-uncased"
SAVE_DIR = "fine_tuned_bert"

TRAIN_FILE = "test_5k.csv"   # 5,000 samples for training
VAL_FILE = "valid_5k.csv"    # 5,000 samples for validation


# =============================================================================
# Helpers
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
# Training
# =============================================================================

def evaluate(model, dataloader, device):
    """Run evaluation and return loss + accuracy."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids, labels=labels)

            total_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)

    return total_loss / len(dataloader), correct / total


def main():
    # --- Load data ---
    print("Loading datasets...")
    train_data = pd.read_csv(TRAIN_FILE).reset_index(drop=True)
    val_data = pd.read_csv(VAL_FILE).reset_index(drop=True)

    label_map = {lbl: i for i, lbl in enumerate(sorted(train_data['label'].unique()))}
    train_data['label_id'] = train_data['label'].map(label_map)
    val_data['label_id'] = val_data['label'].map(label_map)

    print(f"Labels: {label_map}")
    print(f"Train: {len(train_data)} | Val: {len(val_data)}")
    print(f"Train dist:\n{train_data['label'].value_counts().to_string()}")
    print(f"Val dist:\n{val_data['label'].value_counts().to_string()}")

    # --- Device ---
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"Device: {device}")
    else:
        device = torch.device('cpu')
        print(f"Device: {device} (WARNING: training will be slow!)")

    # --- Tokenizer & Model ---
    print(f"\nLoading {MODEL_NAME}...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME, do_lower_case=True)
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(label_map), output_attentions=True
    )
    model = model.to(device)

    # --- Dataloaders ---
    train_dataset = TextDataset(
        train_data['sentence'].values,
        train_data['label_id'].values,
        tokenizer, MAX_LEN
    )
    val_dataset = TextDataset(
        val_data['sentence'].values,
        val_data['label_id'].values,
        tokenizer, MAX_LEN
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- Optimizer & Scheduler ---
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # --- Training loop ---
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)

    best_val_acc = 0.0
    patience = 3
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch in pbar:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids, labels=labels)

            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)

            pbar.set_postfix(loss=f"{total_loss/total:.4f}", acc=f"{correct/total:.4f}")

        scheduler.step()

        train_acc = correct / total
        train_loss = total_loss / len(train_loader)

        # --- Validation ---
        val_loss, val_acc = evaluate(model, val_loader, device)

        print(f"  Epoch {epoch+1}: "
              f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        # --- Early stopping & checkpointing ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            os.makedirs(SAVE_DIR, exist_ok=True)
            model.save_pretrained(SAVE_DIR)
            tokenizer.save_pretrained(SAVE_DIR)
            print(f"  ✓ New best model saved (val_acc={val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break

    # --- Final summary ---
    print("\n" + "=" * 60)
    print(f"TRAINING COMPLETE")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {SAVE_DIR}/")
    print("=" * 60)

    # --- Zip for download ---
    zip_path = f"{SAVE_DIR}.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(SAVE_DIR):
            for file in files:
                filepath = os.path.join(root, file)
                arcname = os.path.relpath(filepath, '.')
                zf.write(filepath, arcname)
    print(f"\nDownloadable zip: {zip_path} ({os.path.getsize(zip_path) / 1e6:.1f} MB)")

    # --- Auto-download on Colab ---
    try:
        from google.colab import files
        files.download(zip_path)
        print("Download started!")
    except ImportError:
        print("Not on Colab — grab the zip manually.")


if __name__ == '__main__':
    main()

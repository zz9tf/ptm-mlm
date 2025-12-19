"""
Script to train the classification head on pre-generated embeddings.
This script loads embeddings and labels, then trains a classification head for site prediction.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from prediction_head import ClassificationHead


class EmbeddingDataset(Dataset):
    """
    Dataset class for embeddings and labels.
    """
    
    def __init__(self, embeddings_list, labels_list):
        """
        Initialize dataset.
        
        @param embeddings_list: List of per-position embeddings (each is a tensor of shape [seq_len, hidden_size])
        @param labels_list: List of label strings (binary strings like "0001000...")
        """
        self.embeddings_list = embeddings_list
        self.labels_list = labels_list
        
        # Convert label strings to tensors
        self.label_tensors = []
        for label_str in labels_list:
            # Convert binary string to tensor of 0s and 1s
            label_tensor = torch.tensor([int(c) for c in label_str], dtype=torch.long)
            self.label_tensors.append(label_tensor)
    
    def __len__(self):
        """Return dataset size."""
        return len(self.embeddings_list)
    
    def __getitem__(self, idx):
        """
        Get a single sample.
        
        @param idx: Sample index
        @returns: Tuple of (embeddings, labels) where both are tensors
        """
        embeddings = self.embeddings_list[idx]  # Shape: [seq_len, hidden_size]
        labels = self.label_tensors[idx]  # Shape: [seq_len]
        
        return embeddings, labels


def collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences.
    
    @param batch: List of (embeddings, labels) tuples
    @returns: Tuple of (padded_embeddings, padded_labels, attention_mask)
    """
    embeddings_list, labels_list = zip(*batch)
    
    # Get max sequence length in batch
    max_len = max(emb.shape[0] for emb in embeddings_list)
    hidden_size = embeddings_list[0].shape[1]
    batch_size = len(batch)
    
    # Initialize padded tensors
    padded_embeddings = torch.zeros(batch_size, max_len, hidden_size)
    padded_labels = torch.zeros(batch_size, max_len, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    
    # Fill in the actual data
    for i, (emb, lab) in enumerate(zip(embeddings_list, labels_list)):
        seq_len = emb.shape[0]
        padded_embeddings[i, :seq_len] = emb
        padded_labels[i, :seq_len] = lab
        attention_mask[i, :seq_len] = True
    
    return padded_embeddings, padded_labels, attention_mask


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train for one epoch.
    
    @param model: Classification head model
    @param dataloader: Data loader
    @param criterion: Loss function
    @param optimizer: Optimizer
    @param device: Device to run on
    @returns: Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_samples = 0
    
    for embeddings, labels, attention_mask in tqdm(dataloader, desc="Training"):
        embeddings = embeddings.to(device)
        labels = labels.to(device)
        attention_mask = attention_mask.to(device)
        
        # Forward pass
        logits = model(embeddings)  # Shape: [batch_size, seq_len, num_labels]
        
        # Reshape for loss calculation
        logits_flat = logits.view(-1, logits.shape[-1])  # [batch_size * seq_len, num_labels]
        labels_flat = labels.view(-1)  # [batch_size * seq_len]
        mask_flat = attention_mask.view(-1)  # [batch_size * seq_len]
        
        # Only compute loss on non-padded positions
        loss = criterion(logits_flat[mask_flat], labels_flat[mask_flat])
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * mask_flat.sum().item()
        num_samples += mask_flat.sum().item()
    
    return total_loss / num_samples if num_samples > 0 else 0.0


def evaluate(model, dataloader, criterion, device):
    """
    Evaluate the model.
    
    @param model: Classification head model
    @param dataloader: Data loader
    @param criterion: Loss function
    @param device: Device to run on
    @returns: Dictionary with metrics
    """
    model.eval()
    total_loss = 0.0
    num_samples = 0
    
    all_predictions = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for embeddings, labels, attention_mask in tqdm(dataloader, desc="Evaluating"):
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            attention_mask = attention_mask.to(device)
            
            # Forward pass
            logits = model(embeddings)  # Shape: [batch_size, seq_len, num_labels]
            
            # Reshape for loss calculation
            logits_flat = logits.view(-1, logits.shape[-1])
            labels_flat = labels.view(-1)
            mask_flat = attention_mask.view(-1)
            
            # Compute loss
            loss = criterion(logits_flat[mask_flat], labels_flat[mask_flat])
            total_loss += loss.item() * mask_flat.sum().item()
            num_samples += mask_flat.sum().item()
            
            # Get predictions
            probs = torch.softmax(logits_flat, dim=-1)
            predictions = torch.argmax(logits_flat, dim=-1)
            
            # Store predictions and labels (only for non-padded positions)
            predictions_batch = predictions[mask_flat].cpu().numpy()
            all_labels_batch = labels_flat[mask_flat].cpu().numpy()
            all_probs_batch = probs[mask_flat, 1].cpu().numpy()  # Probability of positive class
            
            all_predictions.extend(predictions_batch)
            all_labels.extend(all_labels_batch)
            all_probs.extend(all_probs_batch)
    
    # Calculate metrics
    avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='binary', zero_division=0
    )
    
    # Calculate AUC-ROC
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0  # If only one class present
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }


def main():
    parser = argparse.ArgumentParser(description="Train classification head for site prediction")
    parser.add_argument(
        "--embeddings_dir",
        type=str,
        default="/home/zz/zheng/ptm-mlm/downstream_tasks/p_site_prediction/embeddings",
        help="Directory containing pre-generated embeddings"
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=None,
        help="Hidden size of embeddings (will be inferred from embeddings if not provided)"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/zz/zheng/ptm-mlm/downstream_tasks/p_site_prediction",
        help="Output directory for saving trained model"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training"
    )
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device)
    print(f"🔧 Using device: {device}")
    
    # Load embeddings and labels
    print("📦 Loading embeddings and labels...")
    train_embeddings = torch.load(os.path.join(args.embeddings_dir, "train_embeddings.pt"))
    train_labels = torch.load(os.path.join(args.embeddings_dir, "train_labels.pt"))
    train_sequences = torch.load(os.path.join(args.embeddings_dir, "train_sequences.pt"))
    
    print(f"✅ Loaded {len(train_embeddings)} training samples")
    
    # Infer hidden_size from embeddings if not provided
    if args.hidden_size is None or args.hidden_size <= 0:
        # Get hidden_size from first embedding
        if len(train_embeddings) > 0 and len(train_embeddings[0].shape) >= 2:
            inferred_hidden_size = train_embeddings[0].shape[-1]
            print(f"🔍 Inferred hidden_size from embeddings: {inferred_hidden_size}")
            args.hidden_size = inferred_hidden_size
        else:
            raise ValueError("Cannot infer hidden_size from embeddings. Please provide --hidden_size")
    
    # Create dataset and dataloader
    train_dataset = EmbeddingDataset(train_embeddings, train_labels)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Set to 0 to avoid multiprocessing issues
    )
    
    # Initialize model
    print("🚀 Initializing classification head...")
    model = ClassificationHead(
        hidden_size=args.hidden_size,
        num_labels=2,
        dropout=args.dropout
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Training loop
    print("\n" + "="*50)
    print("🎯 Starting training...")
    print("="*50)
    
    best_f1 = 0.0
    best_model_path = os.path.join(args.output_dir, "trained_head.pt")
    
    for epoch in range(args.num_epochs):
        print(f"\n📊 Epoch {epoch + 1}/{args.num_epochs}")
        print("-" * 50)
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"✅ Train Loss: {train_loss:.4f}")
        
        # Evaluate on training set (for monitoring)
        train_metrics = evaluate(model, train_loader, criterion, device)
        print(f"📈 Train Metrics:")
        print(f"   Accuracy: {train_metrics['accuracy']:.4f}")
        print(f"   Precision: {train_metrics['precision']:.4f}")
        print(f"   Recall: {train_metrics['recall']:.4f}")
        print(f"   F1: {train_metrics['f1']:.4f}")
        print(f"   AUC: {train_metrics['auc']:.4f}")
        
        # Save best model
        if train_metrics['f1'] > best_f1:
            best_f1 = train_metrics['f1']
            torch.save({
                'model_state_dict': model.state_dict(),
                'hidden_size': args.hidden_size,
                'num_labels': 2,
                'dropout': args.dropout,
                'epoch': epoch,
                'metrics': train_metrics
            }, best_model_path)
            print(f"💾 Saved best model (F1: {best_f1:.4f}) to {best_model_path}")
    
    print("\n" + "="*50)
    print("🎉 Training completed!")
    print(f"📁 Best model saved to: {best_model_path}")
    print("="*50)


if __name__ == "__main__":
    main()


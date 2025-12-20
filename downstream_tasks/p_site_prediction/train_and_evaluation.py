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
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    confusion_matrix
)
from libauc.losses import AUCMLoss

from prediction_head import ClassificationHead

# 🔧 Fix for PyTorch 2.6: Add numpy as safe global for torch.load
# This is required when loading checkpoints that contain numpy arrays
try:
    import numpy._core.multiarray
    torch.serialization.add_safe_globals([numpy._core.multiarray._reconstruct])
except (ImportError, AttributeError):
    # Fallback for older numpy versions
    try:
        import numpy.core.multiarray
        torch.serialization.add_safe_globals([numpy.core.multiarray._reconstruct])
    except (ImportError, AttributeError):
        pass  # If numpy is not available, skip this step


class CombinedLoss(nn.Module):
    """
    Combined loss function that combines BCEWithLogitsLoss and AUCMLoss.
    Formula: loss = bce_loss + λ * auc_loss
    
    @param pos_weight: Weight for positive class in BCEWithLogitsLoss (for class imbalance)
    @param lambda_weight: Weight (λ) for AUCMLoss (default: 0.5)
    """
    
    def __init__(self, pos_weight=None, lambda_weight=0.5, device=None):
        super().__init__()
        self.lambda_weight = lambda_weight
        self.pos_weight = pos_weight
        self.device = device
        
        # Initialize BCEWithLogitsLoss with pos_weight if provided
        if pos_weight is not None:
            pos_weight_tensor = torch.tensor([pos_weight]).to(device) if device else torch.tensor([pos_weight])
            self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        else:
            self.bce_loss = nn.BCEWithLogitsLoss()
        
        self.auc_loss = AUCMLoss()
    
    def forward(self, binary_logits, binary_labels, probs=None):
        """
        Compute combined loss: loss = bce_loss + λ * auc_loss
        
        @param binary_logits: Binary logits tensor of shape [num_samples] (logits for positive class)
        @param binary_labels: Binary labels tensor of shape [num_samples] (float, 0.0 or 1.0)
        @param probs: Optional probabilities tensor (if None, will compute from binary_logits using sigmoid)
        @returns: Combined loss value
        """
        # Compute BCEWithLogitsLoss
        bce_loss = self.bce_loss(binary_logits, binary_labels)
        
        # Compute AUCMLoss (needs probabilities for positive class)
        if probs is None:
            # Convert logits to probabilities using sigmoid
            probs = torch.sigmoid(binary_logits)
        
        # AUCMLoss expects probabilities and float labels
        auc_loss = self.auc_loss(probs, binary_labels)
        
        # Combine losses: bce_loss + λ * auc_loss
        total_loss = bce_loss + self.lambda_weight * auc_loss
        
        return total_loss


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
        valid_logits = logits_flat[mask_flat]  # [num_valid, num_labels]
        valid_labels = labels_flat[mask_flat]  # [num_valid]
        
        # For binary classification: binary_logits = z1 - z0
        # where z1 is logit for positive class (class 1) and z0 is logit for negative class (class 0)
        binary_logits = valid_logits[:, 1] - valid_logits[:, 0]  # [num_valid] - binary logits
        binary_labels = valid_labels.float()  # [num_valid] - convert to float for BCE
        
        # Compute probabilities for AUC loss (sigmoid of binary logits)
        probs = torch.sigmoid(binary_logits)
        
        # Use combined loss: loss = bce_loss + λ * auc_loss
        loss = criterion(binary_logits, binary_labels, probs=probs)
        
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
            valid_logits = logits_flat[mask_flat]
            valid_labels = labels_flat[mask_flat]
            
            # For binary classification: binary_logits = z1 - z0
            binary_logits = valid_logits[:, 1] - valid_logits[:, 0]  # [num_valid] - binary logits
            binary_labels = valid_labels.float()  # [num_valid] - convert to float for BCE
            
            # Compute probabilities for AUC loss (sigmoid of binary logits)
            probs = torch.sigmoid(binary_logits)
            
            # Use combined loss: loss = bce_loss + λ * auc_loss
            loss = criterion(binary_logits, binary_labels, probs=probs)
            
            total_loss += loss.item() * mask_flat.sum().item()
            num_samples += mask_flat.sum().item()
            
            # Get predictions (for binary classification)
            # binary_logits_all = z1 - z0 for all positions
            binary_logits_all = logits_flat[:, 1] - logits_flat[:, 0]  # [batch_size * seq_len] - binary logits
            binary_probs = torch.sigmoid(binary_logits_all)  # [batch_size * seq_len] - probabilities
            
            # Predictions: 1 if prob > 0.5, else 0
            predictions = (binary_probs > 0.5).long()
            
            # Store predictions and labels (only for non-padded positions)
            predictions_batch = predictions[mask_flat].cpu().numpy()
            all_labels_batch = labels_flat[mask_flat].cpu().numpy()
            all_probs_batch = binary_probs[mask_flat].cpu().numpy()  # Probability of positive class
            
            all_predictions.extend(predictions_batch)
            all_labels.extend(all_labels_batch)
            all_probs.extend(all_probs_batch)
    
    # Calculate metrics
    avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='binary', zero_division=0
    )
    
    # Calculate AUC-ROC (AUROC)
    try:
        auroc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auroc = 0.0  # If only one class present
    
    # Calculate AUPRC (Area Under Precision-Recall Curve)
    try:
        auprc = average_precision_score(all_labels, all_probs)
    except ValueError:
        auprc = 0.0  # If only one class present
    
    # Calculate MCC (Matthews Correlation Coefficient)
    try:
        mcc = matthews_corrcoef(all_labels, all_predictions)
    except ValueError:
        mcc = 0.0
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    # cm format: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auroc': auroc,  # Renamed from 'auc' for clarity
        'auprc': auprc,
        'mcc': mcc,
        'confusion_matrix': cm,
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp)
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
    parser.add_argument(
        "--lambda_weight",
        type=float,
        default=0.5,
        help="Weight (λ) for AUCMLoss in combined loss. Formula: loss = bce_loss + λ * auc_loss (default: 0.5)"
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
    
    # Load validation embeddings and labels if available
    valid_embeddings = None
    valid_labels = None
    valid_emb_path = os.path.join(args.embeddings_dir, "valid_embeddings.pt")
    valid_labels_path = os.path.join(args.embeddings_dir, "valid_labels.pt")
    if os.path.exists(valid_emb_path) and os.path.exists(valid_labels_path):
        valid_embeddings = torch.load(valid_emb_path)
        valid_labels = torch.load(valid_labels_path)
        print(f"✅ Loaded {len(valid_embeddings)} validation samples")
    else:
        print("⚠️  Validation embeddings not found. Will evaluate on training set only.")
    
    # Load test embeddings and labels if available (for final evaluation)
    test_embeddings = None
    test_labels = None
    test_emb_path = os.path.join(args.embeddings_dir, "test_embeddings.pt")
    test_labels_path = os.path.join(args.embeddings_dir, "test_labels.pt")
    if os.path.exists(test_emb_path) and os.path.exists(test_labels_path):
        test_embeddings = torch.load(test_emb_path)
        test_labels = torch.load(test_labels_path)
        print(f"✅ Loaded {len(test_embeddings)} test samples")
    else:
        print("ℹ️  Test embeddings not found. Will skip test evaluation.")
    
    # Infer hidden_size from embeddings if not provided
    if args.hidden_size is None or args.hidden_size <= 0:
        # Get hidden_size from first embedding
        if len(train_embeddings) > 0 and len(train_embeddings[0].shape) >= 2:
            inferred_hidden_size = train_embeddings[0].shape[-1]
            print(f"🔍 Inferred hidden_size from embeddings: {inferred_hidden_size}")
            args.hidden_size = inferred_hidden_size
        else:
            raise ValueError("Cannot infer hidden_size from embeddings. Please provide --hidden_size")
    
    # Create dataset and dataloader for training
    train_dataset = EmbeddingDataset(train_embeddings, train_labels)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Set to 0 to avoid multiprocessing issues
    )
    
    # Create dataset and dataloader for validation if available
    valid_loader = None
    if valid_embeddings is not None and valid_labels is not None:
        valid_dataset = EmbeddingDataset(valid_embeddings, valid_labels)
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            shuffle=False,  # No need to shuffle validation set
            collate_fn=collate_fn,
            num_workers=0
        )
    
    # Create dataset and dataloader for test if available
    test_loader = None
    if test_embeddings is not None and test_labels is not None:
        test_dataset = EmbeddingDataset(test_embeddings, test_labels)
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,  # No need to shuffle test set
            collate_fn=collate_fn,
            num_workers=0
        )
    
    # Initialize model
    print("🚀 Initializing classification head...")
    model = ClassificationHead(
        hidden_size=args.hidden_size,
        num_labels=2,
        dropout=args.dropout
    ).to(device)
    
    # Calculate pos_weight for imbalanced data (for BCEWithLogitsLoss)
    # Count positive and negative samples in training set
    total_positive = 0
    total_negative = 0
    for label_str in train_labels:
        label_array = np.array([int(c) for c in label_str])
        total_positive += label_array.sum()
        total_negative += (label_array == 0).sum()
    
    if total_positive > 0 and total_negative > 0:
        # pos_weight = num_negative / num_positive
        pos_weight = total_negative / total_positive
        print(f"📊 Using Combined Loss (BCEWithLogitsLoss + AUCMLoss) for imbalanced data:")
        print(f"   Positive samples: {total_positive:,}")
        print(f"   Negative samples: {total_negative:,}")
        print(f"   pos_weight (for BCE): {pos_weight:.4f}")
        print(f"   λ (for AUC loss): {args.lambda_weight:.4f}")
        print(f"   Formula: loss = bce_loss + {args.lambda_weight:.4f} * auc_loss")
        criterion = CombinedLoss(
            pos_weight=pos_weight,
            lambda_weight=args.lambda_weight,
            device=device
        )
    else:
        print("⚠️  Cannot calculate pos_weight. Using uniform BCEWithLogitsLoss.")
        print(f"   λ (for AUC loss): {args.lambda_weight:.4f}")
        print(f"   Formula: loss = bce_loss + {args.lambda_weight:.4f} * auc_loss")
        criterion = CombinedLoss(
            pos_weight=None,
            lambda_weight=args.lambda_weight,
            device=device
        )
    
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
        
        # Evaluate on validation set if available
        if valid_loader is not None:
            valid_metrics = evaluate(model, valid_loader, criterion, device)
            print(f"📊 Validation Metrics:")
            print(f"   Accuracy: {valid_metrics['accuracy']:.4f}")
            print(f"   Precision: {valid_metrics['precision']:.4f}")
            print(f"   Recall: {valid_metrics['recall']:.4f}")
            print(f"   F1: {valid_metrics['f1']:.4f}")
            print(f"   AUROC: {valid_metrics['auroc']:.4f}")
            print(f"   AUPRC: {valid_metrics['auprc']:.4f}")
            print(f"   MCC: {valid_metrics['mcc']:.4f}")
            print(f"   Confusion Matrix:")
            print(f"      TN: {valid_metrics['tn']}, FP: {valid_metrics['fp']}")
            print(f"      FN: {valid_metrics['fn']}, TP: {valid_metrics['tp']}")
            
            # Save best model based on validation F1
            if valid_metrics['f1'] > best_f1:
                best_f1 = valid_metrics['f1']
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'hidden_size': args.hidden_size,
                    'num_labels': 2,
                    'dropout': args.dropout,
                    'epoch': epoch,
                    'valid_metrics': valid_metrics
                }, best_model_path)
                print(f"💾 Saved best model (Validation F1: {best_f1:.4f}) to {best_model_path}")
        else:
            print("⚠️  No validation set available. Cannot save best model.")
    
    print("\n" + "="*50)
    print("🎉 Training completed!")
    print(f"📁 Best model saved to: {best_model_path}")
    print("="*50)
    
    # Load best model and evaluate on test set
    if test_loader is not None:
        print("\n" + "="*50)
        print("🔮 Evaluating on test set...")
        print("="*50)
        
        # Load best model
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded best model from epoch {checkpoint.get('epoch', 'unknown')}")
        
        # Evaluate on test set
        test_metrics = evaluate(model, test_loader, criterion, device)
        print(f"📊 Test Metrics:")
        print(f"   Loss: {test_metrics['loss']:.4f}")
        print(f"   Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"   Precision: {test_metrics['precision']:.4f}")
        print(f"   Recall: {test_metrics['recall']:.4f}")
        print(f"   F1: {test_metrics['f1']:.4f}")
        print(f"   AUROC: {test_metrics['auroc']:.4f}")
        print(f"   AUPRC: {test_metrics['auprc']:.4f}")
        print(f"   MCC: {test_metrics['mcc']:.4f}")
        print(f"   Confusion Matrix:")
        print(f"      TN: {test_metrics['tn']}, FP: {test_metrics['fp']}")
        print(f"      FN: {test_metrics['fn']}, TP: {test_metrics['tp']}")
        
        # Save test metrics to checkpoint
        checkpoint['test_metrics'] = test_metrics
        torch.save(checkpoint, best_model_path)
        print(f"💾 Test metrics saved to {best_model_path}")
    else:
        print("\n⚠️  Test set not available. Skipping test evaluation.")


if __name__ == "__main__":
    main()


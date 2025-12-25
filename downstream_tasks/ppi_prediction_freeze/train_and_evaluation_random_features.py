"""
Test script to evaluate performance using random numeric features instead of embeddings.
This script tests the baseline performance when using only numeric features (random numbers)
instead of actual protein sequence embeddings for PPI prediction.
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

from prediction_head import PPIClassificationHead


# Standard 20 amino acids + special tokens
AMINO_ACIDS = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 
                'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
SPECIAL_TOKENS = ['<UNK>', '<PAD>', '<MASK>']
ALL_TOKENS = SPECIAL_TOKENS + AMINO_ACIDS
AA_TO_IDX = {aa: idx for idx, aa in enumerate(ALL_TOKENS)}
IDX_TO_AA = {idx: aa for aa, idx in AA_TO_IDX.items()}
VOCAB_SIZE = len(ALL_TOKENS)
UNK_IDX = AA_TO_IDX['<UNK>']
PAD_IDX = AA_TO_IDX['<PAD>']


class AminoAcidEmbeddingModel(nn.Module):
    """
    Model that uses amino acid embedding layer + classification head.
    This is a baseline model that learns amino acid representations from scratch for PPI prediction.
    """
    
    def __init__(self, vocab_size, hidden_size=1280, num_labels=2, dropout=0.1, embedding_dim=None, pooling_method='mean'):
        """
        Initialize the model.
        
        @param vocab_size: Size of vocabulary (number of amino acids + special tokens)
        @param hidden_size: Hidden size for classification head (default: 1280 to match ESM2)
        @param num_labels: Number of output classes (default: 2 for binary classification)
        @param dropout: Dropout rate (default: 0.1)
        @param embedding_dim: Dimension of amino acid embeddings (default: hidden_size)
        @param pooling_method: Pooling method for sequence-level representation
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim if embedding_dim is not None else hidden_size
        
        # Embedding layer: maps amino acid indices to dense vectors
        self.embedding = nn.Embedding(vocab_size, self.embedding_dim, padding_idx=PAD_IDX)
        
        # Classification head with pooling
        self.classifier = PPIClassificationHead(
            hidden_size=self.embedding_dim,
            num_labels=num_labels,
            dropout=dropout,
            pooling_method=pooling_method
        )
    
    def forward(self, aa_indices, attention_mask=None):
        """
        Forward pass.
        
        @param aa_indices: Amino acid indices tensor of shape (batch_size, seq_len)
        @param attention_mask: Attention mask tensor of shape (batch_size, seq_len)
        @returns: Logits tensor of shape (batch_size, num_labels)
        """
        # Get embeddings from amino acid indices
        # Shape: (batch_size, seq_len) -> (batch_size, seq_len, embedding_dim)
        embeddings = self.embedding(aa_indices)
        
        # Pass through classification head with pooling
        # Shape: (batch_size, seq_len, embedding_dim) -> (batch_size, num_labels)
        logits = self.classifier(embeddings, attention_mask=attention_mask)
        
        return logits


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


class AminoAcidDataset(Dataset):
    """
    Dataset class that converts sequences to amino acid indices and uses embedding layer.
    This is used to test baseline performance without pre-trained embeddings for PPI prediction.
    """
    
    def __init__(self, labels_list, sequences_list):
        """
        Initialize dataset with amino acid indices.
        
        @param labels_list: List of binary labels (0 or 1) for sequence-level classification
        @param sequences_list: List of sequence strings
        """
        self.labels_list = labels_list
        self.sequences_list = sequences_list
        
        # Convert labels to tensors
        self.label_tensors = torch.tensor(labels_list, dtype=torch.long)
        
        # Convert sequences to amino acid indices
        self.aa_indices = []
        for seq in sequences_list:
            # Convert each amino acid to its index
            indices = [AA_TO_IDX.get(aa.upper(), UNK_IDX) for aa in seq]
            self.aa_indices.append(torch.tensor(indices, dtype=torch.long))
    
    def __len__(self):
        """Return dataset size."""
        return len(self.labels_list)
    
    def __getitem__(self, idx):
        """
        Get a single sample with amino acid indices.
        
        @param idx: Sample index
        @returns: Tuple of (aa_indices, label) where aa_indices is a tensor and label is a scalar
        """
        aa_indices = self.aa_indices[idx]  # Shape: [seq_len]
        label = self.label_tensors[idx]  # Scalar: 0 or 1
        
        return aa_indices, label


def collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences.
    
    @param batch: List of (aa_indices, label) tuples
    @returns: Tuple of (padded_indices, labels, attention_mask)
    """
    indices_list, labels_list = zip(*batch)
    
    # Get max sequence length in batch
    max_len = max(idx.shape[0] for idx in indices_list)
    batch_size = len(batch)
    
    # Initialize padded tensors
    padded_indices = torch.full((batch_size, max_len), PAD_IDX, dtype=torch.long)
    labels = torch.tensor(labels_list, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    
    # Fill in the actual data
    for i, idx in enumerate(indices_list):
        seq_len = idx.shape[0]
        padded_indices[i, :seq_len] = idx
        attention_mask[i, :seq_len] = True
    
    return padded_indices, labels, attention_mask


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train for one epoch.
    
    @param model: Model with embedding layer and classification head
    @param dataloader: Data loader
    @param criterion: Loss function
    @param optimizer: Optimizer
    @param device: Device to run on
    @returns: Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_samples = 0
    
    for aa_indices, labels, attention_mask in tqdm(dataloader, desc="Training"):
        aa_indices = aa_indices.to(device)
        labels = labels.to(device)
        attention_mask = attention_mask.to(device)

        # Forward pass
        logits = model(aa_indices, attention_mask=attention_mask)  # Shape: [batch_size, num_labels]
        
        # For binary classification: binary_logits = z1 - z0
        binary_logits = logits[:, 1] - logits[:, 0]  # [batch_size] - binary logits
        binary_labels = labels.float()  # [batch_size] - convert to float for BCE
        
        # Compute probabilities for AUC loss (sigmoid of binary logits)
        probs = torch.sigmoid(binary_logits)
        
        # Use combined loss: loss = bce_loss + λ * auc_loss
        loss = criterion(binary_logits, binary_labels, probs=probs)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(labels)
        num_samples += len(labels)
    
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
        for aa_indices, labels, attention_mask in tqdm(dataloader, desc="Evaluating"):
            aa_indices = aa_indices.to(device)
            labels = labels.to(device)
            attention_mask = attention_mask.to(device)
            
            # Forward pass
            logits = model(aa_indices, attention_mask=attention_mask)  # Shape: [batch_size, num_labels]
            
            # For binary classification: binary_logits = z1 - z0
            binary_logits = logits[:, 1] - logits[:, 0]  # [batch_size] - binary logits
            binary_labels = labels.float()  # [batch_size] - convert to float for BCE
            
            # Compute probabilities for AUC loss (sigmoid of binary logits)
            probs = torch.sigmoid(binary_logits)
            
            # Use combined loss: loss = bce_loss + λ * auc_loss
            loss = criterion(binary_logits, binary_labels, probs=probs)
            
            total_loss += loss.item() * len(labels)
            num_samples += len(labels)
            
            # Predictions: 1 if prob > 0.5, else 0
            predictions = (probs > 0.5).long()
            
            # Store predictions and labels
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
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
        'auroc': auroc,
        'auprc': auprc,
        'mcc': mcc,
        'confusion_matrix': cm,
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp)
    }


def main():
    parser = argparse.ArgumentParser(description="Train classification head with amino acid embeddings (baseline test) for PPI prediction")
    parser.add_argument(
        "--embeddings_dir",
        type=str,
        default="/home/zz/zheng/ptm-mlm/downstream_tasks/ppi_prediction/embeddings",
        help="Directory containing labels and sequences (for loading data)"
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=1280,
        help="Hidden size of random features (should match embedding hidden_size, default: 1280 for ESM2)"
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=None,
        help="Dimension of amino acid embeddings (default: same as hidden_size)"
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
        "--pooling_method",
        type=str,
        default="mean",
        choices=["mean", "max", "cls", "attention"],
        help="Pooling method for sequence-level representation (default: mean)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/zz/zheng/ptm-mlm/downstream_tasks/ppi_prediction",
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
    print(f"📏 Hidden size: {args.hidden_size}")
    print(f"📚 Vocabulary size: {VOCAB_SIZE} (20 amino acids + 3 special tokens)")
    if args.embedding_dim:
        print(f"🔢 Embedding dimension: {args.embedding_dim}")
    else:
        print(f"🔢 Embedding dimension: {args.hidden_size} (same as hidden_size)")
    
    # Load labels and sequences (we need sequences to get lengths)
    print("📦 Loading labels and sequences...")
    train_labels = torch.load(os.path.join(args.embeddings_dir, "train_labels.pt"))
    train_sequences = torch.load(os.path.join(args.embeddings_dir, "train_sequences.pt"))
    
    print(f"✅ Loaded {len(train_labels)} training samples")
    
    # Load validation labels and sequences if available
    valid_labels = None
    valid_sequences = None
    valid_labels_path = os.path.join(args.embeddings_dir, "valid_labels.pt")
    valid_sequences_path = os.path.join(args.embeddings_dir, "valid_sequences.pt")
    if os.path.exists(valid_labels_path) and os.path.exists(valid_sequences_path):
        valid_labels = torch.load(valid_labels_path)
        valid_sequences = torch.load(valid_sequences_path)
        print(f"✅ Loaded {len(valid_labels)} validation samples")
    else:
        print("⚠️  Validation data not found. Will evaluate on training set only.")
    
    # Load test labels and sequences if available (for final evaluation)
    test_labels = None
    test_sequences = None
    test_labels_path = os.path.join(args.embeddings_dir, "test_labels.pt")
    test_sequences_path = os.path.join(args.embeddings_dir, "test_sequences.pt")
    if os.path.exists(test_labels_path) and os.path.exists(test_sequences_path):
        test_labels = torch.load(test_labels_path)
        test_sequences = torch.load(test_sequences_path)
        print(f"✅ Loaded {len(test_labels)} test samples")
    else:
        print("ℹ️  Test data not found. Will skip test evaluation.")
    
    # Create dataset and dataloader for training
    train_dataset = AminoAcidDataset(train_labels, train_sequences)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Set to 0 to avoid multiprocessing issues
    )
    
    # Create dataset and dataloader for validation if available
    valid_loader = None
    if valid_labels is not None and valid_sequences is not None:
        valid_dataset = AminoAcidDataset(valid_labels, valid_sequences)
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            shuffle=False,  # No need to shuffle validation set
            collate_fn=collate_fn,
            num_workers=0
        )
    
    # Create dataset and dataloader for test if available
    test_loader = None
    if test_labels is not None and test_sequences is not None:
        test_dataset = AminoAcidDataset(test_labels, test_sequences)
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,  # No need to shuffle test set
            collate_fn=collate_fn,
            num_workers=0
        )
    
    # Initialize model
    print("🚀 Initializing model with amino acid embedding layer...")
    embedding_dim = args.embedding_dim if args.embedding_dim else args.hidden_size
    model = AminoAcidEmbeddingModel(
        vocab_size=VOCAB_SIZE,
        hidden_size=args.hidden_size,
        num_labels=2,
        dropout=args.dropout,
        embedding_dim=embedding_dim,
        pooling_method=args.pooling_method
    ).to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Calculate pos_weight for imbalanced data (for BCEWithLogitsLoss)
    # Count positive and negative samples in training set
    total_positive = sum(train_labels)
    total_negative = len(train_labels) - total_positive
    
    if total_positive > 0 and total_negative > 0:
        # pos_weight = num_negative / num_positive
        pos_weight = total_negative / total_positive
        print(f"📊 Using Combined Loss (BCEWithLogitsLoss + AUCMLoss) for imbalanced data:")
        print(f"   Positive samples (Enhance): {total_positive:,}")
        print(f"   Negative samples (Inhibit): {total_negative:,}")
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
    print("🎯 Starting training with amino acid embedding model...")
    print("="*50)
    
    best_f1 = 0.0
    output_filename = "trained_head_aa_embedding.pt"
    best_model_path = os.path.join(args.output_dir, output_filename)
    
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
                    'vocab_size': VOCAB_SIZE,
                    'hidden_size': args.hidden_size,
                    'embedding_dim': embedding_dim,
                    'num_labels': 2,
                    'dropout': args.dropout,
                    'pooling_method': args.pooling_method,
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
        
        # Print summary
        print("\n" + "="*50)
        print("📈 BASELINE TEST SUMMARY (Amino Acid Embedding Model)")
        print("="*50)
        print(f"   Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"   Test F1: {test_metrics['f1']:.4f}")
        print(f"   Test AUROC: {test_metrics['auroc']:.4f}")
        print(f"   Test AUPRC: {test_metrics['auprc']:.4f}")
        print(f"   Test MCC: {test_metrics['mcc']:.4f}")
        print("="*50)
    else:
        print("\n⚠️  Test set not available. Skipping test evaluation.")


if __name__ == "__main__":
    main()















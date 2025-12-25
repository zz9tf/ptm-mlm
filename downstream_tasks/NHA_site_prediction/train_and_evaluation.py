"""
Script to train the classification head on pre-generated embeddings for NHA site prediction.
This script loads embeddings and labels, then trains a classification head for sequence-level binary classification.
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
import json
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    confusion_matrix
)
from libauc.losses import AUCMLoss

from GNNTrans import GNNTrans
from torch_geometric.data import Data, Batch


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
        
        @param binary_logits: Binary logits tensor of shape [batch_size] (logits for positive class)
        @param binary_labels: Binary labels tensor of shape [batch_size] (float, 0.0 or 1.0)
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
    Dataset class for per-position embeddings, labels, and sequences.
    For NHA site prediction, each amino acid position is a node in the graph.
    """
    
    def __init__(self, embeddings_list, labels_list, sequences_list=None):
        """
        Initialize dataset.
        
        @param embeddings_list: List of per-position embeddings (each is a tensor of shape [seq_len, hidden_size])
        @param labels_list: List of integer labels (0 or 1)
        @param sequences_list: List of protein sequences (strings). If None, will use dummy sequences.
        """
        self.embeddings_list = embeddings_list
        self.labels_list = labels_list
        self.sequences_list = sequences_list
        
        # Convert labels to tensors
        self.label_tensors = torch.tensor(labels_list, dtype=torch.long)
    
    def __len__(self):
        """Return dataset size."""
        return len(self.embeddings_list)
    
    def __getitem__(self, idx):
        """
        Get a single sample.
        
        @param idx: Sample index
        @returns: Tuple of (embeddings, label, sequence) where embeddings is [seq_len, hidden_size], label is a scalar, sequence is a string
        """
        embeddings = self.embeddings_list[idx]  # Shape: [seq_len, hidden_size]
        label = self.label_tensors[idx]  # Scalar (0 or 1)
        
        # Get sequence if available, otherwise use dummy
        if self.sequences_list is not None:
            sequence = self.sequences_list[idx]
        else:
            raise ValueError("Sequences not found. Please provide sequences.")
        
        return embeddings, label, sequence


def collate_fn(batch):
    """
    Collate function for per-position embeddings, converting to graph format for GNNTrans.
    Each amino acid position is a node, connected in a chain: 0-1, 1-2, 2-3, ...
    
    @param batch: List of (embeddings, label, sequence) tuples
    @returns: Tuple of (graph_batch, labels_tensor)
    """
    embeddings_list, labels_list, sequences_list = zip(*batch)
    
    # Create graph data for each sample
    # Each sample is a chain graph where each amino acid position is a node
    graph_list = []
    for emb, label, sequence in zip(embeddings_list, labels_list, sequences_list):
        # emb shape: [seq_len, hidden_size]
        # Each row is a node (amino acid position)
        x = emb  # [seq_len, hidden_size] - seq_len nodes
        
        # Create chain edges: 0-1, 1-2, 2-3, ..., (seq_len-2)-(seq_len-1)
        # Edge format: [[source_nodes], [target_nodes]]
        seq_len = x.shape[0]
        if seq_len > 1:
            # Forward edges: i -> i+1
            source_nodes = torch.arange(seq_len - 1, dtype=torch.long)
            target_nodes = torch.arange(1, seq_len, dtype=torch.long)
            # Also add backward edges for bidirectional: i+1 -> i
            edge_index = torch.stack([
                torch.cat([source_nodes, target_nodes]),  # [0,1,2,...,n-2, 1,2,3,...,n-1]
                torch.cat([target_nodes, source_nodes])   # [1,2,3,...,n-1, 0,1,2,...,n-2]
            ])
        else:
            raise ValueError("Single node not supported")
        
        # Use the real sequence data
        # GNNTrans uses data.seq[0] to find center position: idx = (data.ptr + int(len(data.seq[0]) / 2))[:-1]
        # This will select the middle node of each sequence
        seq = [sequence]  # Real sequence data
        
        data = Data(
            x=x,
            edge_index=edge_index,
            seq=seq
        )
        graph_list.append(data)
    
    # Batch all graphs together
    # Batch.from_data_list automatically creates batch, ptr attributes
    graph_batch = Batch.from_data_list(graph_list)
    
    # Ensure emb attribute exists (GNNTrans expects data.emb)
    # In PyG, x is the node features, but GNNTrans uses data.emb
    graph_batch.emb = graph_batch.x
    
    # Stack labels: [batch_size]
    labels_tensor = torch.stack([torch.tensor(l, dtype=torch.long) for l in labels_list])
    
    return graph_batch, labels_tensor


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train for one epoch.
    
    @param model: GNNTrans model
    @param dataloader: Data loader
    @param criterion: Loss function
    @param optimizer: Optimizer
    @param device: Device to run on
    @returns: Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_samples = 0
    
    for graph_batch, labels in tqdm(dataloader, desc="Training"):
        # Move graph data to device
        graph_batch = graph_batch.to(device)
        labels = labels.to(device)

        # Forward pass through GNNTrans
        # GNNTrans outputs sigmoid probabilities [batch_size, 1]
        probs = model(graph_batch)  # Shape: [batch_size, 1]
        probs = probs.squeeze(1)  # Shape: [batch_size]
        
        # Convert probabilities to logits for BCEWithLogitsLoss
        # logit = log(prob / (1 - prob))
        # To avoid numerical issues, use inverse sigmoid
        eps = 1e-8
        probs_clamped = torch.clamp(probs, eps, 1 - eps)
        binary_logits = torch.log(probs_clamped / (1 - probs_clamped))
        
        binary_labels = labels.float()  # [batch_size] - convert to float for BCE
        
        # Use combined loss: loss = bce_loss + λ * auc_loss
        loss = criterion(binary_logits, binary_labels, probs=probs)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(labels)
        num_samples += len(labels)
    
    return total_loss / num_samples if num_samples > 0 else 0.0


def plot_training_curves(training_history, save_path):
    """
    Plot training curves and save to file.
    
    @param training_history: Dictionary containing training history
    @param save_path: Path to save the plot
    """
    epochs = range(1, len(training_history['train_loss']) + 1)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Curves', fontsize=16, fontweight='bold')
    
    # Plot 1: Loss
    ax1 = axes[0, 0]
    ax1.plot(epochs, training_history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    if any(v is not None for v in training_history['valid_loss']):
        valid_loss = [v if v is not None else np.nan for v in training_history['valid_loss']]
        ax1.plot(epochs, valid_loss, 'r-', label='Valid Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss Curves', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: F1 Score
    ax2 = axes[0, 1]
    if any(v is not None for v in training_history['valid_f1']):
        valid_f1 = [v if v is not None else np.nan for v in training_history['valid_f1']]
        ax2.plot(epochs, valid_f1, 'g-', label='Valid F1', linewidth=2, marker='o', markersize=4)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('F1 Score', fontsize=12)
    ax2.set_title('F1 Score', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: AUROC and AUPRC
    ax3 = axes[1, 0]
    if any(v is not None for v in training_history['valid_auroc']):
        valid_auroc = [v if v is not None else np.nan for v in training_history['valid_auroc']]
        ax3.plot(epochs, valid_auroc, 'm-', label='Valid AUROC', linewidth=2, marker='s', markersize=4)
    if any(v is not None for v in training_history['valid_auprc']):
        valid_auprc = [v if v is not None else np.nan for v in training_history['valid_auprc']]
        ax3.plot(epochs, valid_auprc, 'c-', label='Valid AUPRC', linewidth=2, marker='^', markersize=4)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Score', fontsize=12)
    ax3.set_title('AUROC and AUPRC', fontsize=14)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Accuracy, Precision, Recall
    ax4 = axes[1, 1]
    if any(v is not None for v in training_history['valid_accuracy']):
        valid_acc = [v if v is not None else np.nan for v in training_history['valid_accuracy']]
        ax4.plot(epochs, valid_acc, 'orange', label='Valid Accuracy', linewidth=2, marker='o', markersize=4)
    if any(v is not None for v in training_history['valid_precision']):
        valid_prec = [v if v is not None else np.nan for v in training_history['valid_precision']]
        ax4.plot(epochs, valid_prec, 'purple', label='Valid Precision', linewidth=2, marker='s', markersize=4)
    if any(v is not None for v in training_history['valid_recall']):
        valid_rec = [v if v is not None else np.nan for v in training_history['valid_recall']]
        ax4.plot(epochs, valid_rec, 'brown', label='Valid Recall', linewidth=2, marker='^', markersize=4)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Score', fontsize=12)
    ax4.set_title('Accuracy, Precision, Recall', fontsize=14)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def evaluate(model, dataloader, criterion, device):
    """
    Evaluate the model.
    
    @param model: GNNTrans model
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
        for graph_batch, labels in tqdm(dataloader, desc="Evaluating"):
            # Move graph data to device
            graph_batch = graph_batch.to(device)
            labels = labels.to(device)
            
            # Forward pass through GNNTrans
            # GNNTrans outputs sigmoid probabilities [batch_size, 1]
            probs = model(graph_batch)  # Shape: [batch_size, 1]
            probs = probs.squeeze(1)  # Shape: [batch_size]
            
            # Convert probabilities to logits for BCEWithLogitsLoss
            eps = 1e-8
            probs_clamped = torch.clamp(probs, eps, 1 - eps)
            binary_logits = torch.log(probs_clamped / (1 - probs_clamped))
            
            binary_labels = labels.float()  # [batch_size] - convert to float for BCE
            
            # Use combined loss: loss = bce_loss + λ * auc_loss
            loss = criterion(binary_logits, binary_labels, probs=probs)
            
            total_loss += loss.item() * len(labels)
            num_samples += len(labels)
            
            # Get predictions
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
    parser = argparse.ArgumentParser(description="Train classification head for NHA site prediction")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/zz/zheng/ptm-mlm/downstream_tasks/outputs/NHA_site_prediction",
        help="Output directory. Embeddings will be loaded from output_dir/embeddings/"
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
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"📁 Output directory: {args.output_dir}")
    
    # Automatically find embeddings directory in output_dir
    embeddings_dir = os.path.join(args.output_dir, "embeddings")
    if not os.path.exists(embeddings_dir):
        raise FileNotFoundError(
            f"Embeddings directory not found: {embeddings_dir}\n"
            f"Please run generate_embeddings.py first to generate embeddings in {embeddings_dir}"
        )
    print(f"📁 Embeddings directory: {embeddings_dir}")
    
    # Set device
    device = torch.device(args.device)
    print(f"🔧 Using device: {device}")
    
    # Load embeddings, labels, and sequences
    print("📦 Loading embeddings, labels, and sequences...")
    train_embeddings = torch.load(os.path.join(embeddings_dir, "train_embeddings.pt"), weights_only=False)
    train_labels = torch.load(os.path.join(embeddings_dir, "train_labels.pt"), weights_only=False)
    
    # Load sequences if available
    train_sequences = None
    train_seqs_path = os.path.join(embeddings_dir, "train_sequences.pt")
    if os.path.exists(train_seqs_path):
        train_sequences = torch.load(train_seqs_path, weights_only=False)
        print(f"✅ Loaded {len(train_embeddings)} training samples (with sequences)")
    else:
        print(f"✅ Loaded {len(train_embeddings)} training samples (sequences not found, will use dummy)")
    
    # Load validation embeddings, labels, and sequences if available
    valid_embeddings = None
    valid_labels = None
    valid_sequences = None
    valid_emb_path = os.path.join(embeddings_dir, "valid_embeddings.pt")
    valid_labels_path = os.path.join(embeddings_dir, "valid_labels.pt")
    valid_seqs_path = os.path.join(embeddings_dir, "valid_sequences.pt")
    if os.path.exists(valid_emb_path) and os.path.exists(valid_labels_path):
        valid_embeddings = torch.load(valid_emb_path, weights_only=False)
        valid_labels = torch.load(valid_labels_path, weights_only=False)
        if os.path.exists(valid_seqs_path):
            valid_sequences = torch.load(valid_seqs_path, weights_only=False)
        print(f"✅ Loaded {len(valid_embeddings)} validation samples")
    else:
        print("⚠️  Validation embeddings not found. Cannot save best model based on validation metrics.")
    
    # Load test embeddings, labels, and sequences if available
    test_embeddings = None
    test_labels = None
    test_sequences = None
    test_emb_path = os.path.join(embeddings_dir, "test_embeddings.pt")
    test_labels_path = os.path.join(embeddings_dir, "test_labels.pt")
    test_seqs_path = os.path.join(embeddings_dir, "test_sequences.pt")
    if os.path.exists(test_emb_path) and os.path.exists(test_labels_path):
        test_embeddings = torch.load(test_emb_path, weights_only=False)
        test_labels = torch.load(test_labels_path, weights_only=False)
        if os.path.exists(test_seqs_path):
            test_sequences = torch.load(test_seqs_path, weights_only=False)
        print(f"✅ Loaded {len(test_embeddings)} test samples")
    else:
        print("ℹ️  Test embeddings not found. Will skip test evaluation.")
    
    # Infer hidden_size from embeddings if not provided
    if args.hidden_size is None or args.hidden_size <= 0:
        # Get hidden_size from first embedding
        # embeddings are now per-position: [seq_len, hidden_size]
        if len(train_embeddings) > 0:
            if isinstance(train_embeddings[0], torch.Tensor):
                # Shape is [seq_len, hidden_size], so hidden_size is the last dimension
                if len(train_embeddings[0].shape) >= 2:
                    inferred_hidden_size = train_embeddings[0].shape[-1]
                else:
                    inferred_hidden_size = train_embeddings[0].shape[0]
            else:
                inferred_hidden_size = len(train_embeddings[0])
            print(f"🔍 Inferred hidden_size from embeddings: {inferred_hidden_size}")
            args.hidden_size = inferred_hidden_size
        else:
            raise ValueError("Cannot infer hidden_size from embeddings. Please provide --hidden_size")
    
    # Create dataset and dataloader for training
    train_dataset = EmbeddingDataset(train_embeddings, train_labels, train_sequences)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Create dataset and dataloader for validation if available
    valid_loader = None
    if valid_embeddings is not None and valid_labels is not None:
        valid_dataset = EmbeddingDataset(valid_embeddings, valid_labels, valid_sequences)
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )
    
    # Create dataset and dataloader for test if available
    test_loader = None
    if test_embeddings is not None and test_labels is not None:
        test_dataset = EmbeddingDataset(test_embeddings, test_labels, test_sequences)
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )
    
    # Initialize GNNTrans model
    print("🚀 Initializing GNNTrans model...")
    # GNNTrans parameters
    hidden_dim = 128  # Hidden dimension for GNN layers
    num_layers = 3   # Number of GNN layers
    
    model = GNNTrans(
        input_dim=args.hidden_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers
    ).to(device)
    
    print(f"   Input dim: {args.hidden_size}")
    print(f"   Hidden dim: {hidden_dim}")
    print(f"   Num layers: {num_layers}")
    
    # Calculate pos_weight for imbalanced data (for BCEWithLogitsLoss)
    # Count positive and negative samples in training set
    total_positive = sum(train_labels)
    total_negative = len(train_labels) - total_positive
    
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
    
    # Training history for saving results
    training_history = {
        'train_loss': [],
        'valid_loss': [],
        'valid_accuracy': [],
        'valid_precision': [],
        'valid_recall': [],
        'valid_f1': [],
        'valid_auroc': [],
        'valid_auprc': [],
        'valid_mcc': []
    }
    
    for epoch in range(args.num_epochs):
        print(f"\n📊 Epoch {epoch + 1}/{args.num_epochs}")
        print("-" * 50)
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"✅ Train Loss: {train_loss:.4f}")
        training_history['train_loss'].append(float(train_loss))
        
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
            
            # Record validation metrics
            training_history['valid_loss'].append(float(valid_metrics['loss']))
            training_history['valid_accuracy'].append(float(valid_metrics['accuracy']))
            training_history['valid_precision'].append(float(valid_metrics['precision']))
            training_history['valid_recall'].append(float(valid_metrics['recall']))
            training_history['valid_f1'].append(float(valid_metrics['f1']))
            training_history['valid_auroc'].append(float(valid_metrics['auroc']))
            training_history['valid_auprc'].append(float(valid_metrics['auprc']))
            training_history['valid_mcc'].append(float(valid_metrics['mcc']))
            
            # Save best model based on validation F1
            if valid_metrics['f1'] > best_f1:
                best_f1 = valid_metrics['f1']
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'input_dim': args.hidden_size,
                    'hidden_dim': hidden_dim,
                    'num_layers': num_layers,
                    'epoch': epoch,
                    'valid_metrics': valid_metrics
                }, best_model_path)
                print(f"💾 Saved best model (Validation F1: {best_f1:.4f}) to {best_model_path}")
        else:
            print("⚠️  No validation set available. Cannot save best model based on validation metrics.")
            # Fill with None for epochs without validation
            training_history['valid_loss'].append(None)
            training_history['valid_accuracy'].append(None)
            training_history['valid_precision'].append(None)
            training_history['valid_recall'].append(None)
            training_history['valid_f1'].append(None)
            training_history['valid_auroc'].append(None)
            training_history['valid_auprc'].append(None)
            training_history['valid_mcc'].append(None)
    
    # If no validation set, save the final model anyway
    if valid_loader is None:
        print("\n💾 Saving final model (no validation set available)...")
        torch.save({
            'model_state_dict': model.state_dict(),
            'input_dim': args.hidden_size,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'epoch': args.num_epochs - 1,
            'note': 'Final model saved without validation metrics'
        }, best_model_path)
        print(f"💾 Final model saved to {best_model_path}")
    
    print("\n" + "="*50)
    print("🎉 Training completed!")
    print(f"📁 Model saved to: {best_model_path}")
    print("="*50)
    
    # Save training history to JSON
    history_path = os.path.join(args.output_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    print(f"💾 Training history saved to {history_path}")
    
    # Plot and save training curves
    try:
        plot_path = os.path.join(args.output_dir, "training_curves.png")
        plot_training_curves(training_history, plot_path)
        print(f"📊 Training curves saved to {plot_path}")
    except Exception as e:
        print(f"⚠️  Failed to plot training curves: {e}")
    
    # Load best model and evaluate on test set
    test_metrics = None
    if test_loader is not None:
        print("\n" + "="*50)
        print("🔮 Evaluating on test set...")
        print("="*50)
        
        # Load best model
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        # Recreate model with saved parameters
        saved_input_dim = checkpoint.get('input_dim', args.hidden_size)
        saved_hidden_dim = checkpoint.get('hidden_dim', hidden_dim)
        saved_num_layers = checkpoint.get('num_layers', num_layers)
        model = GNNTrans(
            input_dim=saved_input_dim,
            hidden_dim=saved_hidden_dim,
            num_layers=saved_num_layers
        ).to(device)
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
    
    # Save final evaluation results to JSON
    results = {
        'training_config': {
            'num_epochs': args.num_epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'dropout': args.dropout,
            'lambda_weight': args.lambda_weight,
            'hidden_size': args.hidden_size,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers
        },
        'best_validation_f1': float(best_f1),
        'training_history': training_history
    }
    
    # Add test metrics if available
    if test_metrics is not None:
        results['test_metrics'] = {
            'loss': float(test_metrics['loss']),
            'accuracy': float(test_metrics['accuracy']),
            'precision': float(test_metrics['precision']),
            'recall': float(test_metrics['recall']),
            'f1': float(test_metrics['f1']),
            'auroc': float(test_metrics['auroc']),
            'auprc': float(test_metrics['auprc']),
            'mcc': float(test_metrics['mcc']),
            'confusion_matrix': {
                'tn': int(test_metrics['tn']),
                'fp': int(test_metrics['fp']),
                'fn': int(test_metrics['fn']),
                'tp': int(test_metrics['tp'])
            }
        }
    
    # Add best validation metrics if available
    if valid_loader is not None and len(training_history['valid_f1']) > 0:
        best_epoch = np.argmax(training_history['valid_f1'])
        results['best_validation_metrics'] = {
            'epoch': int(best_epoch + 1),
            'loss': training_history['valid_loss'][best_epoch],
            'accuracy': training_history['valid_accuracy'][best_epoch],
            'precision': training_history['valid_precision'][best_epoch],
            'recall': training_history['valid_recall'][best_epoch],
            'f1': training_history['valid_f1'][best_epoch],
            'auroc': training_history['valid_auroc'][best_epoch],
            'auprc': training_history['valid_auprc'][best_epoch],
            'mcc': training_history['valid_mcc'][best_epoch]
        }
    
    results_path = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"💾 Final evaluation results saved to {results_path}")


if __name__ == "__main__":
    main()


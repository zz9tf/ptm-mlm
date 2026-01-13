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
import sys
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

from prediction_head import ClassificationHead

# Add project root and downstream_tasks directory to sys.path for importing modules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..', '..')
downstream_tasks_dir = os.path.join(current_dir, '..', '..')

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if downstream_tasks_dir not in sys.path:
    sys.path.insert(0, downstream_tasks_dir)

# Import inference pipeline for dynamic embedding processing
from utils.inference.inference_pipeline import InferencePipeline


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
        "--output_dir",
        type=str,
        default="/home/zz/zheng/ptm-mlm/downstream_tasks/outputs/p_site_prediction",
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
        help="Batch size for training and adaptor processing. "
             "This parameter controls both the training DataLoader batch size and "
             "the batch size used when processing embeddings through adaptor."
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
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Model name for dynamic embedding generation (e.g., 'EvolutionaryScale_esmc-600m-2024-12'). "
             "When provided along with --layer_index and --adaptor_checkpoint, embeddings will be "
             "loaded from embeddings/ directory and processed through adaptor."
    )
    parser.add_argument(
        "--layer_index",
        type=int,
        default=None,
        help="Layer index for dynamic embedding generation (e.g., 30). "
             "Used together with --model_name and --adaptor_checkpoint."
    )
    parser.add_argument(
        "--adaptor_checkpoint",
        type=str,
        default=None,
        help="Adaptor checkpoint name (without .ckpt extension) for processing embeddings. "
             "When provided with --model_name and --layer_index, enables dynamic embedding processing."
    )

    args = parser.parse_args()
    
    # 🔧 Convert "None" string to None for adaptor_checkpoint
    if args.adaptor_checkpoint is not None:
        if args.adaptor_checkpoint.lower() in ['none', 'null', '']:
            args.adaptor_checkpoint = None
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"📁 Output directory: {args.output_dir}")

    # Set device
    device = torch.device(args.device)
    print(f"🔧 Using device: {device}")

    # Check if using new dynamic embedding processing
    # 🔧 Allow adaptor_checkpoint to be None (means no adaptor processing)
    use_dynamic_processing = (args.model_name is not None and
                            args.layer_index is not None)

    if use_dynamic_processing:
        print("🔄 Using dynamic embedding processing...")
        print(f"   Model: {args.model_name}, Layer: {args.layer_index}")
        if args.adaptor_checkpoint is not None:
            print(f"   Adaptor checkpoint: {args.adaptor_checkpoint}")
        else:
            print(f"   Adaptor checkpoint: None (using raw embeddings)")

        # Initialize inference pipeline
        pipeline = InferencePipeline()

        # Prepare data using inference pipeline (with progress bar)
        print("🔄 Preparing data for p_site training...")
        processed_data = pipeline.prepare_data_for_training(
            model_name=args.model_name,
            layer_index=args.layer_index,
            task_name="p_site",
            adaptor_checkpoint=args.adaptor_checkpoint,
            device=args.device,
            batch_size=args.batch_size
        )

        # Extract processed embeddings and labels
        train_embeddings = processed_data['train_embeddings']
        train_labels = processed_data['train_labels']
        valid_embeddings = processed_data.get('valid_embeddings')
        valid_labels = processed_data.get('valid_labels')
        test_embeddings = processed_data.get('test_embeddings')
        test_labels = processed_data.get('test_labels')

        # Load sequences separately (not processed through adaptor)
        embeddings_base_path = pipeline.get_embeddings_path(args.model_name, args.layer_index) / "p_site"
        print(f"✅ Loaded {len(train_embeddings)} training samples")

        if valid_embeddings is not None:
            print(f"✅ Loaded {len(valid_embeddings)} validation samples")
        if test_embeddings is not None:
            print(f"✅ Loaded {len(test_embeddings)} test samples")
    else:
        print("📦 Using legacy embedding loading from output_dir...")
        # Legacy logic: load from output_dir/embeddings
        embeddings_dir = os.path.join(args.output_dir, "embeddings")
        if not os.path.exists(embeddings_dir):
            raise FileNotFoundError(
                f"Embeddings directory not found: {embeddings_dir}\n"
                f"Please run generate_embeddings.py first to generate embeddings in {embeddings_dir}\n"
                f"Or use --model_name, --layer_index, and --adaptor_checkpoint for dynamic processing."
            )
        print(f"📁 Embeddings directory: {embeddings_dir}")

        # Load embeddings and labels
        print("📦 Loading embeddings and labels...")
        train_embeddings = torch.load(os.path.join(embeddings_dir, "train_embeddings.pt"), weights_only=False)
        train_labels = torch.load(os.path.join(embeddings_dir, "train_labels.pt"), weights_only=False)

        print(f"✅ Loaded {len(train_embeddings)} training samples")

        # Load validation embeddings and labels if available
        valid_embeddings = None
        valid_labels = None
        valid_emb_path = os.path.join(embeddings_dir, "valid_embeddings.pt")
        valid_labels_path = os.path.join(embeddings_dir, "valid_labels.pt")
        if os.path.exists(valid_emb_path) and os.path.exists(valid_labels_path):
            valid_embeddings = torch.load(valid_emb_path, weights_only=False)
            valid_labels = torch.load(valid_labels_path, weights_only=False)
            print(f"✅ Loaded {len(valid_embeddings)} validation samples")
        else:
            print("⚠️  Validation embeddings not found. Will evaluate on training set only.")

        # Load test embeddings and labels if available (for final evaluation)
        test_embeddings = None
        test_labels = None
        test_emb_path = os.path.join(embeddings_dir, "test_embeddings.pt")
        test_labels_path = os.path.join(embeddings_dir, "test_labels.pt")
        if os.path.exists(test_emb_path) and os.path.exists(test_labels_path):
            test_embeddings = torch.load(test_emb_path, weights_only=False)
            test_labels = torch.load(test_labels_path, weights_only=False)
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
                    'hidden_size': args.hidden_size,
                    'num_labels': 2,
                    'dropout': args.dropout,
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
            'hidden_size': args.hidden_size,
            'num_labels': 2,
            'dropout': args.dropout,
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
            'hidden_size': args.hidden_size
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


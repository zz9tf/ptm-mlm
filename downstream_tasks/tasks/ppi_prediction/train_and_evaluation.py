"""
Script to train the TransformerClassifier on pre-generated embeddings for PPI prediction.
This script loads embeddings (binder, wt, ptm) and labels, then trains a classification model.
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
from ppi import TransformerClassifier


class CombinedLoss(nn.Module):
    """
    Loss function using BCEWithLogitsLoss only.
    
    @param pos_weight: Weight for positive class in BCEWithLogitsLoss (for class imbalance)
    @param device: Device to run on
    """
    
    def __init__(self, pos_weight=None, device=None):
        super().__init__()
        self.pos_weight = pos_weight
        self.device = device
        
        # Initialize BCEWithLogitsLoss with pos_weight if provided
        if pos_weight is not None:
            pos_weight_tensor = torch.tensor([pos_weight]).to(device) if device else torch.tensor([pos_weight])
            self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        else:
            self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, binary_logits, binary_labels):
        """
        Compute BCE loss.
        
        @param binary_logits: Binary logits tensor of shape [batch_size] (logits for positive class)
        @param binary_labels: Binary labels tensor of shape [batch_size] (float, 0.0 or 1.0)
        @returns: BCE loss value
        """
        # Compute BCEWithLogitsLoss
        loss = self.bce_loss(binary_logits, binary_labels)
        
        return loss


class PPIDataset(Dataset):
    """
    Dataset class for PPI prediction with binder, wt, and ptm embeddings.
    """
    
    def __init__(self, binder_embeddings, wt_embeddings, ptm_embeddings, labels):
        """
        Initialize dataset.
        
        @param binder_embeddings: List of binder embeddings (each is a tensor of shape [hidden_size])
        @param wt_embeddings: List of wild-type embeddings (each is a tensor of shape [hidden_size])
        @param ptm_embeddings: List of PTM-modified embeddings (each is a tensor of shape [hidden_size])
        @param labels: List of integer labels (0 or 1)
        """
        self.binder_embeddings = binder_embeddings
        self.wt_embeddings = wt_embeddings
        self.ptm_embeddings = ptm_embeddings
        self.labels = labels
        
        # Convert labels to tensors
        self.label_tensors = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        """Return dataset size."""
        return len(self.binder_embeddings)
    
    def __getitem__(self, idx):
        """
        Get a single sample.
        
        @param idx: Sample index
        @returns: Tuple of (binder_emb, wt_emb, ptm_emb, label)
        """
        binder_emb = self.binder_embeddings[idx]  # Shape: [hidden_size]
        wt_emb = self.wt_embeddings[idx]  # Shape: [hidden_size]
        ptm_emb = self.ptm_embeddings[idx]  # Shape: [hidden_size]
        label = self.label_tensors[idx]  # Scalar (0 or 1)
        
        return binder_emb, wt_emb, ptm_emb, label


def collate_fn(batch):
    """
    Collate function for PPI dataset.
    
    @param batch: List of (binder_emb, wt_emb, ptm_emb, label) tuples
    @returns: Tuple of (binder_embs, wt_embs, ptm_embs, labels)
    """
    binder_embs, wt_embs, ptm_embs, labels = zip(*batch)
    
    # Stack embeddings and labels
    binder_embs = torch.stack(binder_embs)  # [batch_size, hidden_size]
    wt_embs = torch.stack(wt_embs)  # [batch_size, hidden_size]
    ptm_embs = torch.stack(ptm_embs)  # [batch_size, hidden_size]
    labels = torch.stack(labels)  # [batch_size]
    
    return binder_embs, wt_embs, ptm_embs, labels


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train for one epoch.
    
    @param model: TransformerClassifier model
    @param dataloader: Data loader
    @param criterion: Loss function
    @param optimizer: Optimizer
    @param device: Device to run on
    @returns: Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_samples = 0
    
    for binder_embs, wt_embs, ptm_embs, labels in tqdm(dataloader, desc="Training"):
        # Move data to device
        binder_embs = binder_embs.to(device)
        wt_embs = wt_embs.to(device)
        ptm_embs = ptm_embs.to(device)
        labels = labels.to(device)
        
        # Forward pass
        logits = model(binder_embs, wt_embs, ptm_embs)  # Shape: [batch_size]
        
        # Convert labels to float for BCE loss
        binary_labels = labels.float()
        
        # Compute loss (BCE only, no AUC loss)
        loss = criterion(logits, binary_labels)
        
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
    
    @param model: TransformerClassifier model
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
        for binder_embs, wt_embs, ptm_embs, labels in tqdm(dataloader, desc="Evaluating"):
            # Move data to device
            binder_embs = binder_embs.to(device)
            wt_embs = wt_embs.to(device)
            ptm_embs = ptm_embs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            logits = model(binder_embs, wt_embs, ptm_embs)  # Shape: [batch_size]
            
            # Convert labels to float for BCE loss
            binary_labels = labels.float()
            
            # Compute loss (BCE only, no AUC loss)
            loss = criterion(logits, binary_labels)
            
            # Compute probabilities for evaluation metrics (AUROC, AUPRC, etc.)
            probs = torch.sigmoid(logits)
            
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
    parser = argparse.ArgumentParser(description="Train TransformerClassifier for PPI prediction")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/zz/zheng/ptm-mlm/downstream_tasks/outputs/ppi_prediction_freeze",
        help="Output directory. Embeddings will be loaded from output_dir/embeddings/"
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=None,
        help="Hidden size of embeddings (will be inferred from embeddings if not provided)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2000,
        help="Maximum length for TransformerClassifier (default: 2000)"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=50,
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
        default=0.3,
        help="Dropout rate"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training"
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
    
    # Load embeddings and labels
    print("📦 Loading embeddings and labels...")
    
    # Load training data
    train_binder_embeddings = torch.load(os.path.join(embeddings_dir, "train_binder_embeddings.pt"), weights_only=False)
    train_wt_embeddings = torch.load(os.path.join(embeddings_dir, "train_wt_embeddings.pt"), weights_only=False)
    train_ptm_embeddings = torch.load(os.path.join(embeddings_dir, "train_ptm_embeddings.pt"), weights_only=False)
    train_labels = torch.load(os.path.join(embeddings_dir, "train_labels.pt"), weights_only=False)
    print(f"✅ Loaded {len(train_labels)} training samples")
    
    # Load validation data if available
    valid_binder_embeddings = None
    valid_wt_embeddings = None
    valid_ptm_embeddings = None
    valid_labels = None
    valid_binder_path = os.path.join(embeddings_dir, "valid_binder_embeddings.pt")
    if os.path.exists(valid_binder_path):
        valid_binder_embeddings = torch.load(valid_binder_path, weights_only=False)
        valid_wt_embeddings = torch.load(os.path.join(embeddings_dir, "valid_wt_embeddings.pt"), weights_only=False)
        valid_ptm_embeddings = torch.load(os.path.join(embeddings_dir, "valid_ptm_embeddings.pt"), weights_only=False)
        valid_labels = torch.load(os.path.join(embeddings_dir, "valid_labels.pt"), weights_only=False)
        print(f"✅ Loaded {len(valid_labels)} validation samples")
    else:
        print("⚠️  Validation embeddings not found. Cannot save best model based on validation metrics.")
    
    # Load test data if available
    test_binder_embeddings = None
    test_wt_embeddings = None
    test_ptm_embeddings = None
    test_labels = None
    test_binder_path = os.path.join(embeddings_dir, "test_binder_embeddings.pt")
    if os.path.exists(test_binder_path):
        test_binder_embeddings = torch.load(test_binder_path, weights_only=False)
        test_wt_embeddings = torch.load(os.path.join(embeddings_dir, "test_wt_embeddings.pt"), weights_only=False)
        test_ptm_embeddings = torch.load(os.path.join(embeddings_dir, "test_ptm_embeddings.pt"), weights_only=False)
        test_labels = torch.load(os.path.join(embeddings_dir, "test_labels.pt"), weights_only=False)
        print(f"✅ Loaded {len(test_labels)} test samples")
    else:
        print("ℹ️  Test embeddings not found. Will skip test evaluation.")
    
    # Infer hidden_size from embeddings if not provided
    if args.hidden_size is None or args.hidden_size <= 0:
        if len(train_binder_embeddings) > 0:
            if isinstance(train_binder_embeddings[0], torch.Tensor):
                inferred_hidden_size = train_binder_embeddings[0].shape[-1]
            else:
                inferred_hidden_size = len(train_binder_embeddings[0])
            print(f"🔍 Inferred hidden_size from embeddings: {inferred_hidden_size}")
            args.hidden_size = inferred_hidden_size
        else:
            raise ValueError("Cannot infer hidden_size from embeddings. Please provide --hidden_size")
    
    # Create dataset and dataloader for training
    train_dataset = PPIDataset(train_binder_embeddings, train_wt_embeddings, train_ptm_embeddings, train_labels)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    # Create dataset and dataloader for validation if available
    valid_dataloader = None
    if valid_binder_embeddings is not None:
        valid_dataset = PPIDataset(valid_binder_embeddings, valid_wt_embeddings, valid_ptm_embeddings, valid_labels)
        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )
    
    # Create dataset and dataloader for test if available
    test_dataloader = None
    if test_binder_embeddings is not None:
        test_dataset = PPIDataset(test_binder_embeddings, test_wt_embeddings, test_ptm_embeddings, test_labels)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )
    
    # Initialize model
    print(f"\n🏗️  Initializing TransformerClassifier...")
    print(f"   Hidden size: {args.hidden_size}")
    print(f"   Max length: {args.max_length}")
    print(f"   Dropout: {args.dropout}")
    
    # TransformerClassifier expects [batch_size, max_length] inputs
    # But we have [batch_size, hidden_size] embeddings
    # Need to project hidden_size to max_length
    class EmbeddingProjector(nn.Module):
        """Project embeddings from hidden_size to max_length."""
        def __init__(self, hidden_size, max_length):
            super().__init__()
            self.projection = nn.Linear(hidden_size, max_length)
        
        def forward(self, x):
            # x: [batch_size, hidden_size]
            return self.projection(x)  # [batch_size, max_length]
    
    projector = EmbeddingProjector(args.hidden_size, args.max_length).to(device)
    model = TransformerClassifier(dropout_rate=args.dropout, max_length=args.max_length).to(device)
    
    # Wrap model with projector
    class ProjectedModel(nn.Module):
        """Wrapper to project embeddings before passing to TransformerClassifier."""
        def __init__(self, projector, model):
            super().__init__()
            self.projector = projector
            self.model = model
        
        def forward(self, binder, wt, ptm):
            # Project embeddings to max_length
            binder = self.projector(binder)  # [batch_size, max_length]
            wt = self.projector(wt)  # [batch_size, max_length]
            ptm = self.projector(ptm)  # [batch_size, max_length]
            return self.model(binder, wt, ptm)
    
    model = ProjectedModel(projector, model).to(device)
    
    # Calculate class weights for imbalanced data
    pos_count = sum(train_labels)
    neg_count = len(train_labels) - pos_count
    if pos_count > 0 and neg_count > 0:
        pos_weight = neg_count / pos_count
    else:
        pos_weight = None
    
    # Initialize loss function
    criterion = CombinedLoss(pos_weight=pos_weight, device=device)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Training history
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
    
    # Best model tracking
    best_valid_f1 = 0.0
    best_model_path = os.path.join(args.output_dir, "best_model.pt")
    
    # Training loop
    print(f"\n🚀 Starting training for {args.num_epochs} epochs...")
    for epoch in range(1, args.num_epochs + 1):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch}/{args.num_epochs}")
        print(f"{'='*70}")
        
        # Train
        train_loss = train_epoch(model, train_dataloader, criterion, optimizer, device)
        training_history['train_loss'].append(train_loss)
        print(f"📊 Train Loss: {train_loss:.4f}")
        
        # Validate
        if valid_dataloader is not None:
            valid_metrics = evaluate(model, valid_dataloader, criterion, device)
            training_history['valid_loss'].append(valid_metrics['loss'])
            training_history['valid_accuracy'].append(valid_metrics['accuracy'])
            training_history['valid_precision'].append(valid_metrics['precision'])
            training_history['valid_recall'].append(valid_metrics['recall'])
            training_history['valid_f1'].append(valid_metrics['f1'])
            training_history['valid_auroc'].append(valid_metrics['auroc'])
            training_history['valid_auprc'].append(valid_metrics['auprc'])
            training_history['valid_mcc'].append(valid_metrics['mcc'])
            
            print(f"📊 Valid Loss: {valid_metrics['loss']:.4f}")
            print(f"📊 Valid Accuracy: {valid_metrics['accuracy']:.4f}")
            print(f"📊 Valid Precision: {valid_metrics['precision']:.4f}")
            print(f"📊 Valid Recall: {valid_metrics['recall']:.4f}")
            print(f"📊 Valid F1: {valid_metrics['f1']:.4f}")
            print(f"📊 Valid AUROC: {valid_metrics['auroc']:.4f}")
            print(f"📊 Valid AUPRC: {valid_metrics['auprc']:.4f}")
            print(f"📊 Valid MCC: {valid_metrics['mcc']:.4f}")
            
            # Save best model based on validation F1
            if valid_metrics['f1'] > best_valid_f1:
                best_valid_f1 = valid_metrics['f1']
                torch.save(model.state_dict(), best_model_path)
                print(f"✅ Saved best model (F1: {best_valid_f1:.4f})")
        else:
            training_history['valid_loss'].append(None)
            training_history['valid_accuracy'].append(None)
            training_history['valid_precision'].append(None)
            training_history['valid_recall'].append(None)
            training_history['valid_f1'].append(None)
            training_history['valid_auroc'].append(None)
            training_history['valid_auprc'].append(None)
            training_history['valid_mcc'].append(None)
    
    # Load best model for final evaluation
    if os.path.exists(best_model_path):
        print(f"\n📦 Loading best model from {best_model_path}...")
        model.load_state_dict(torch.load(best_model_path))
    
    # Final evaluation on test set
    if test_dataloader is not None:
        print(f"\n{'='*70}")
        print("🧪 Final Evaluation on Test Set")
        print(f"{'='*70}")
        test_metrics = evaluate(model, test_dataloader, criterion, device)
        print(f"📊 Test Loss: {test_metrics['loss']:.4f}")
        print(f"📊 Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"📊 Test Precision: {test_metrics['precision']:.4f}")
        print(f"📊 Test Recall: {test_metrics['recall']:.4f}")
        print(f"📊 Test F1: {test_metrics['f1']:.4f}")
        print(f"📊 Test AUROC: {test_metrics['auroc']:.4f}")
        print(f"📊 Test AUPRC: {test_metrics['auprc']:.4f}")
        print(f"📊 Test MCC: {test_metrics['mcc']:.4f}")
        print(f"📊 Confusion Matrix:")
        print(f"   TN: {test_metrics['tn']}, FP: {test_metrics['fp']}")
        print(f"   FN: {test_metrics['fn']}, TP: {test_metrics['tp']}")
        
        # Save test metrics
        test_metrics_path = os.path.join(args.output_dir, "test_metrics.json")
        test_metrics_save = {k: v for k, v in test_metrics.items() if k != 'confusion_matrix'}
        test_metrics_save['confusion_matrix'] = test_metrics['confusion_matrix'].tolist()
        with open(test_metrics_path, 'w') as f:
            json.dump(test_metrics_save, f, indent=2)
        print(f"✅ Saved test metrics to {test_metrics_path}")
    
    # Save training history
    history_path = os.path.join(args.output_dir, "training_history.json")
    training_history_save = {k: [v if v is not None else None for v in vs] for k, vs in training_history.items()}
    with open(history_path, 'w') as f:
        json.dump(training_history_save, f, indent=2)
    print(f"✅ Saved training history to {history_path}")
    
    # Plot training curves
    plot_path = os.path.join(args.output_dir, "training_curves.png")
    plot_training_curves(training_history, plot_path)
    print(f"✅ Saved training curves to {plot_path}")
    
    print("\n" + "="*70)
    print("🎉 Training completed!")
    print("="*70)


if __name__ == "__main__":
    main()


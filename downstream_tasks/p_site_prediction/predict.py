"""
Script to make predictions on test data using trained classification head.
"""
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import argparse
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix

from prediction_head import ClassificationHead


class EmbeddingDataset(Dataset):
    """
    Dataset class for embeddings (without labels for prediction).
    """
    
    def __init__(self, embeddings_list):
        """
        Initialize dataset.
        
        @param embeddings_list: List of per-position embeddings (each is a tensor of shape [seq_len, hidden_size])
        """
        self.embeddings_list = embeddings_list
    
    def __len__(self):
        """Return dataset size."""
        return len(self.embeddings_list)
    
    def __getitem__(self, idx):
        """
        Get a single sample.
        
        @param idx: Sample index
        @returns: Embeddings tensor
        """
        return self.embeddings_list[idx]


def collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences.
    
    @param batch: List of embeddings tensors
    @returns: Tuple of (padded_embeddings, attention_mask, original_lengths)
    """
    embeddings_list = batch
    
    # Get max sequence length in batch
    max_len = max(emb.shape[0] for emb in embeddings_list)
    hidden_size = embeddings_list[0].shape[1]
    batch_size = len(batch)
    
    # Initialize padded tensors
    padded_embeddings = torch.zeros(batch_size, max_len, hidden_size)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    original_lengths = []
    
    # Fill in the actual data
    for i, emb in enumerate(embeddings_list):
        seq_len = emb.shape[0]
        padded_embeddings[i, :seq_len] = emb
        attention_mask[i, :seq_len] = True
        original_lengths.append(seq_len)
    
    return padded_embeddings, attention_mask, original_lengths


def predict(model, dataloader, device):
    """
    Make predictions on the dataset.
    
    @param model: Trained classification head model
    @param dataloader: Data loader
    @param device: Device to run on
    @returns: List of predictions (one per sequence, each is a list of per-position predictions)
    """
    model.eval()
    all_predictions = []
    all_probs = []
    
    with torch.no_grad():
        for embeddings, attention_mask, original_lengths in tqdm(dataloader, desc="Predicting"):
            embeddings = embeddings.to(device)
            attention_mask = attention_mask.to(device)
            
            # Forward pass
            logits = model(embeddings)  # Shape: [batch_size, seq_len, num_labels]
            
            # Get predictions and probabilities
            probs = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
            
            # Extract predictions for each sequence (remove padding)
            for i, (pred, prob, mask, orig_len) in enumerate(zip(predictions, probs, attention_mask, original_lengths)):
                # Get only non-padded positions
                pred_seq = pred[mask].cpu().numpy()
                prob_seq = prob[mask, 1].cpu().numpy()  # Probability of positive class
                
                all_predictions.append(pred_seq)
                all_probs.append(prob_seq)
    
    return all_predictions, all_probs


def evaluate_predictions(predictions, labels):
    """
    Evaluate predictions against ground truth labels.
    
    @param predictions: List of prediction arrays (one per sequence)
    @param labels: List of label strings (binary strings)
    @returns: Dictionary with metrics
    """
    # Flatten predictions and labels
    all_preds = []
    all_labels = []
    
    for pred, label_str in zip(predictions, labels):
        label_array = np.array([int(c) for c in label_str])
        # Ensure same length
        min_len = min(len(pred), len(label_array))
        all_preds.extend(pred[:min_len])
        all_labels.extend(label_array[:min_len])
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }


def save_predictions(predictions, sequences, output_path):
    """
    Save predictions to file.
    
    @param predictions: List of prediction arrays
    @param sequences: List of sequences
    @param output_path: Path to save predictions
    """
    with open(output_path, 'w') as f:
        f.write("seq,prediction\n")
        for seq, pred in zip(sequences, predictions):
            # Convert prediction array to binary string
            pred_str = ''.join([str(p) for p in pred])
            f.write(f"{seq},{pred_str}\n")
    print(f"✅ Predictions saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Make predictions on test data")
    parser.add_argument(
        "--embeddings_dir",
        type=str,
        default="/home/zz/zheng/ptm-mlm/downstream_tasks/p_site_prediction/embeddings",
        help="Directory containing pre-generated embeddings"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/zz/zheng/ptm-mlm/downstream_tasks/p_site_prediction/trained_head.pt",
        help="Path to trained classification head"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for prediction"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/home/zz/zheng/ptm-mlm/downstream_tasks/p_site_prediction/test_predictions.txt",
        help="Output path for predictions"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for prediction"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate predictions if test labels are available"
    )
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device)
    print(f"🔧 Using device: {device}")
    
    # Load embeddings
    print("📦 Loading test embeddings...")
    test_embeddings = torch.load(os.path.join(args.embeddings_dir, "test_embeddings.pt"))
    test_sequences = torch.load(os.path.join(args.embeddings_dir, "test_sequences.pt"))
    
    print(f"✅ Loaded {len(test_embeddings)} test samples")
    
    # Create dataset and dataloader
    test_dataset = EmbeddingDataset(test_embeddings)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Load trained model
    print(f"📦 Loading trained model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Get model parameters from checkpoint
    hidden_size = checkpoint.get('hidden_size')
    num_labels = checkpoint.get('num_labels', 2)
    dropout = checkpoint.get('dropout', 0.1)
    
    # If hidden_size is not in checkpoint, infer from embeddings
    if hidden_size is None:
        if len(test_embeddings) > 0 and len(test_embeddings[0].shape) >= 2:
            hidden_size = test_embeddings[0].shape[-1]
            print(f"🔍 Inferred hidden_size from embeddings: {hidden_size}")
        else:
            raise ValueError("Cannot infer hidden_size. Please ensure checkpoint contains 'hidden_size' or embeddings are valid")
    
    model = ClassificationHead(
        hidden_size=hidden_size,
        num_labels=num_labels,
        dropout=dropout
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✅ Model loaded successfully!")
    
    # Make predictions
    print("\n" + "="*50)
    print("🔮 Making predictions...")
    print("="*50)
    predictions, probabilities = predict(model, test_loader, device)
    
    # Save predictions
    save_predictions(predictions, test_sequences, args.output_path)
    
    # Evaluate if labels are available
    if args.evaluate:
        print("\n" + "="*50)
        print("📊 Evaluating predictions...")
        print("="*50)
        test_labels = torch.load(os.path.join(args.embeddings_dir, "test_labels.pt"))
        metrics = evaluate_predictions(predictions, test_labels)
        
        print(f"📈 Test Metrics:")
        print(f"   Accuracy: {metrics['accuracy']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall: {metrics['recall']:.4f}")
        print(f"   F1: {metrics['f1']:.4f}")
        print(f"\n📊 Confusion Matrix:")
        print(metrics['confusion_matrix'])
    
    print("\n" + "="*50)
    print("🎉 Prediction completed!")
    print("="*50)


if __name__ == "__main__":
    main()


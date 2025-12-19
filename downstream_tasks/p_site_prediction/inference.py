"""
Model inference script for generating embeddings from pre-trained Mamba model.
This script loads a trained checkpoint and generates embeddings for protein sequences.
"""
import torch
import sys
import os
from pathlib import Path
from accelerate import Accelerator
from tqdm import tqdm

# Add main_pipeline to path
main_pipeline_path = Path(__file__).parent.parent.parent / "main_pipeline"
sys.path.insert(0, str(main_pipeline_path))

from getters.tokenizer import PTMTokenizer
from utils.checkpoint import load_ckpt
from models.mamba.lm import MambaLMHeadModel


class ModelInference:
    """
    Model inference class for generating embeddings from pre-trained Mamba model.
    """
    
    def __init__(self, checkpoint_path: str, device: str = None):
        """
        Initialize the inference model.
        
        @param checkpoint_path: Path to the trained model checkpoint (.ckpt file)
        @param device: Device to run inference on (None for auto-detect)
        """
        self.accelerator = Accelerator()
        if device:
            self.accelerator.device = torch.device(device)
        
        # Load tokenizer
        self.tokenizer = PTMTokenizer()
        
        # Load model from checkpoint
        print(f"📦 Loading model from {checkpoint_path}...")
        self.model = load_ckpt(
            checkpoint_path, 
            self.tokenizer, 
            self.accelerator
        )
        self.model.eval()
        
        # Get hidden size from model config
        self.hidden_size = self.model.config.d_model
        print(f"✅ Model loaded successfully! Hidden size: {self.hidden_size}")
    
    @torch.no_grad()
    def generate_embeddings(self, sequences: list, batch_size: int = 32, return_pooled: bool = False):
        """
        Generate embeddings for a list of sequences.
        
        @param sequences: List of protein sequences (strings)
        @param batch_size: Batch size for inference
        @param return_pooled: If True, return pooled embeddings (mean pooling). 
                             If False, return sequence-level embeddings (last token or mean)
        @returns: Tensor of embeddings with shape (num_sequences, hidden_size) or 
                 (num_sequences, seq_len, hidden_size) if not pooled
        """
        all_embeddings = []
        
        # Process in batches
        for i in tqdm(range(0, len(sequences), batch_size), desc="Generating embeddings"):
            batch_sequences = sequences[i:i + batch_size]
            
            # Tokenize sequences
            input_ids = self.tokenizer(
                batch_sequences,
                add_special_tokens=True,
                return_tensors=True
            ).to(self.accelerator.device)
            
            # Generate embeddings using model backbone
            # The backbone returns hidden states for all tokens
            hidden_states = self.model.backbone(input_ids)
            
            if return_pooled:
                # Mean pooling over sequence length (excluding padding)
                # Get attention mask (non-padding tokens)
                attention_mask = (input_ids != self.tokenizer.pad_token_id).float()
                # Compute mean pooling
                pooled_embeddings = (hidden_states * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
                all_embeddings.append(pooled_embeddings.cpu())
            else:
                # Return all token embeddings (can be used for per-position prediction)
                all_embeddings.append(hidden_states.cpu())
        
        # Concatenate all batches
        embeddings = torch.cat(all_embeddings, dim=0)
        return embeddings
    
    @torch.no_grad()
    def generate_per_position_embeddings(self, sequences: list, batch_size: int = 32):
        """
        Generate per-position embeddings for sequences (useful for site prediction).
        Returns embeddings for each position in the sequence.
        
        @param sequences: List of protein sequences (strings)
        @param batch_size: Batch size for inference
        @returns: List of tensors, each with shape (seq_len, hidden_size)
        """
        all_embeddings = []
        original_lengths = []
        
        # Process in batches
        for i in tqdm(range(0, len(sequences), batch_size), desc="Generating per-position embeddings"):
            batch_sequences = sequences[i:i + batch_size]
            original_lengths.extend([len(seq) for seq in batch_sequences])
            
            # Tokenize sequences
            input_ids = self.tokenizer(
                batch_sequences,
                add_special_tokens=True,
                return_tensors=True
            ).to(self.accelerator.device)
            
            # Generate embeddings using model backbone
            hidden_states = self.model.backbone(input_ids)
            
            # Remove special tokens (CLS and EOS) and padding
            # Note: The tokenizer tokenizes character by character, so each character becomes one token
            # CLS is at position 0, EOS is at the end before padding
            batch_input_ids = input_ids.cpu()
            eos_token_id = self.tokenizer.ids_to_tokens.index("<eos>")
            
            for j, (seq, hidden, input_id) in enumerate(zip(batch_sequences, hidden_states, batch_input_ids)):
                # Find where the actual sequence tokens are (excluding CLS, EOS, and padding)
                # CLS token is at position 0, sequence tokens start at 1
                # Find EOS token position
                eos_pos = None
                for pos in range(1, len(input_id)):
                    if input_id[pos].item() == eos_token_id:
                        eos_pos = pos
                        break
                
                if eos_pos is not None:
                    # Extract embeddings from position 1 to eos_pos (exclusive)
                    # This gives us embeddings for each character in the sequence
                    seq_embeddings = hidden[1:eos_pos]
                else:
                    # Fallback: if no EOS found, use all tokens except CLS
                    # This shouldn't happen, but handle it gracefully
                    seq_embeddings = hidden[1:]
                
                all_embeddings.append(seq_embeddings.cpu())
        
        return all_embeddings, original_lengths


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate embeddings from pre-trained model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--sequences", type=str, nargs="+", help="Input sequences")
    parser.add_argument("--output", type=str, help="Output path to save embeddings")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    
    args = parser.parse_args()
    
    # Initialize inference model
    inferencer = ModelInference(args.checkpoint)
    
    # Generate embeddings
    embeddings = inferencer.generate_embeddings(
        args.sequences, 
        batch_size=args.batch_size,
        return_pooled=True
    )
    
    # Save embeddings if output path is provided
    if args.output:
        torch.save(embeddings, args.output)
        print(f"✅ Embeddings saved to {args.output}")
    else:
        print(f"📊 Generated embeddings shape: {embeddings.shape}")


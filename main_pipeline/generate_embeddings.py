"""
Script to pre-generate ESM embeddings for all sequences in the dataset.
This allows faster training by loading pre-computed embeddings instead of computing them on-the-fly.
Supports multi-GPU generation using accelerate.
"""
import torch
import esm
import argparse
import os
import yaml
from tqdm import tqdm
from pathlib import Path
import h5py
import numpy as np
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from getters.tokenizer import PTMTokenizer
from getters.ptm_dataset import get_ptm_dataset
from utils.config import load_config
from utils.esm_utils import make_esm_input_ids


def compute_extraction_windows(sequence_length, max_length):
    """
    Compute extraction windows for sequences of different lengths.
    
    Strategy:
    - sequence < max_length: Extract once (full sequence)
    - max_length <= sequence < 10 * max_length: Uniform extraction with overlap
    - sequence >= 10 * max_length: Uniform extraction without overlap
    
    @param sequence_length: Length of the sequence
    @param max_length: Maximum sequence length for each window
    @return: List of (start_idx, end_idx) tuples for each window
    """
    if max_length is None or sequence_length <= max_length:
        # Case 1: Sequence is shorter than max_length, extract once
        return [(0, sequence_length)]
    
    if sequence_length < 10 * max_length:
        # Case 2: Medium length sequence, uniform extraction with overlap
        # Calculate number of windows to ensure good coverage
        # Target: ~30% overlap between adjacent windows
        num_windows = int(np.ceil(sequence_length / (max_length * 0.7)))
        num_windows = max(2, num_windows)  # At least 2 windows
        
        # Calculate step size to ensure uniform distribution with overlap
        # Formula: step_size = (sequence_length - max_length) / (num_windows - 1)
        # This ensures first window starts at 0 and last window ends at sequence_length
        if num_windows == 1:
            step_size = 0
        else:
            step_size = (sequence_length - max_length) / (num_windows - 1)
        
        windows = []
        for i in range(num_windows):
            start_idx = int(i * step_size)
            end_idx = min(start_idx + max_length, sequence_length)
            if end_idx > start_idx:  # Ensure valid window
                windows.append((start_idx, end_idx))
        
        # Ensure the last window covers the end of the sequence
        if windows and windows[-1][1] < sequence_length:
            windows[-1] = (max(0, sequence_length - max_length), sequence_length)
        
        return windows
    else:
        # Case 3: Long sequence, uniform extraction without overlap
        num_windows = int(np.ceil(sequence_length / max_length))
        windows = []
        for i in range(num_windows):
            start_idx = i * max_length
            end_idx = min(start_idx + max_length, sequence_length)
            if end_idx > start_idx:  # Ensure valid window
                windows.append((start_idx, end_idx))
        return windows


def generate_embeddings_for_dataset(
    dataset,
    tokenizer,
    esm_model,
    batch_converter,
    accelerator,
    output_dir,
    batch_size=8,
    max_sequence_length=None,
):
    """
    Generate ESM embeddings for all sequences in the dataset.
    @param dataset: Dataset dict with train/val/test splits.
    @param tokenizer: Tokenizer instance.
    @param esm_model: ESM model (prepared by accelerator).
    @param batch_converter: ESM alphabet batch converter.
    @param accelerator: Accelerator instance for distributed processing.
    @param output_dir: Directory to save embeddings.
    @param batch_size: Batch size for embedding generation.
    @param max_sequence_length: Maximum sequence length (same as training). If None, no limit.
    """
    device = accelerator.device
    
    # Only create directory on main process
    if accelerator.is_local_main_process:
        os.makedirs(output_dir, exist_ok=True)
    
    # Process each split
    for split_name in ["train", "val", "test"]:
        if split_name not in dataset or dataset[split_name] is None:
            continue
        
        split_dataset = dataset[split_name]
        if accelerator.is_local_main_process:
            accelerator.print(f"\n📊 Processing {split_name} set ({len(split_dataset)} samples)...")
        
        # Collect all embeddings and metadata (only on main process)
        all_embeddings = []
        all_unique_ids = []
        all_lengths = []
        
        # Distribute samples across processes
        num_processes = accelerator.num_processes
        process_rank = accelerator.process_index
        
        # Calculate indices for this process
        total_samples = len(split_dataset)
        samples_per_process = total_samples // num_processes
        start_idx = process_rank * samples_per_process
        end_idx = start_idx + samples_per_process if process_rank < num_processes - 1 else total_samples
        
        # Process in batches
        process_indices = range(start_idx, end_idx)
        if accelerator.is_local_main_process:
            pbar = tqdm(range(0, len(process_indices), batch_size), desc=f"Generating {split_name} embeddings")
        else:
            pbar = range(0, len(process_indices), batch_size)
        
        for batch_offset in pbar:
            batch_samples = []
            batch_indices = []
            
            batch_start = start_idx + batch_offset
            batch_end = min(batch_start + batch_size, end_idx)
            
            for j in range(batch_start, batch_end):
                sample = split_dataset[j]
                batch_samples.append(sample)
                batch_indices.append(j)
            
            # Get sequences and unique_ids
            sequences = [s["sequence"] for s in batch_samples]
            unique_ids = [s.get("unique_id", f"{split_name}_{j}") for j, s in zip(batch_indices, batch_samples)]
            
            # Tokenize sequences
            tokenized = tokenizer(
                sequences,
                add_special_tokens=True,
                max_sequence_length=None,  # Don't truncate at tokenization, we'll crop later
            )
            
            # Convert to tensors and prepare for ESM
            input_ids_list = [torch.tensor(ids, device=device) for ids in tokenized]
            
            # Process each sequence with window extraction strategy
            # Store each window separately (not merged) for random selection during training
            for idx, input_ids in enumerate(input_ids_list):
                seq_len = len(input_ids)
                unique_id = unique_ids[idx]
                
                # Compute extraction windows for this sequence
                windows = compute_extraction_windows(seq_len, max_sequence_length)
                
                # Process each window separately
                for win_idx, (window_start, window_end) in enumerate(windows):
                    window_ids = input_ids[window_start:window_end]
                    window_len = window_end - window_start
                    
                    # Prepare ESM input (ESM batch_converter will handle padding)
                    window_ids = window_ids.unsqueeze(0)  # Add batch dimension
                    esm_input_ids = make_esm_input_ids(window_ids, tokenizer)
                    
                    # Convert to sequence for ESM
                    esm_input = (0, tokenizer.decode(esm_input_ids[0].detach().cpu().tolist()))
                    
                    # Compute ESM embedding
                    with torch.no_grad():
                        batch_labels, batch_strs, batch_tokens = batch_converter([esm_input])
                        batch_tokens = batch_tokens[..., 1:-1].to(device)  # remove <cls> and <eos>
                        out = esm_model(batch_tokens, repr_layers=[33], return_contacts=False)
                        window_emb = out["representations"][33][0]  # Shape: (seq_len, embed_dim)
                    
                    # Extract actual window embedding (remove padding if any)
                    window_emb_actual = window_emb[:window_len].cpu().numpy()
                    
                    # Store each window separately with unique identifier
                    if len(windows) == 1:
                        # Single window: use original unique_id
                        window_unique_id = unique_id
                    else:
                        # Multiple windows: append window index to unique_id
                        window_unique_id = f"{unique_id}_window_{win_idx}"
                    
                    all_embeddings.append(window_emb_actual)
                    all_unique_ids.append(window_unique_id)
                    all_lengths.append(window_len)
        
        # Gather all embeddings from all processes
        accelerator.wait_for_everyone()
        
        # Use gather_object to collect Python objects (lists) from all processes
        from accelerate.utils import gather_object
        
        # Gather all data from all processes
        gathered_unique_ids = gather_object(all_unique_ids) if all_unique_ids else []
        gathered_lengths = gather_object(all_lengths) if all_lengths else []
        gathered_embeddings = gather_object(all_embeddings) if all_embeddings else []
        
        # Flatten nested lists (gather_object returns a list of lists, one per process)
        if gathered_unique_ids and isinstance(gathered_unique_ids[0], list):
            gathered_unique_ids = [item for sublist in gathered_unique_ids for item in sublist]
        if gathered_lengths and isinstance(gathered_lengths[0], list):
            gathered_lengths = [item for sublist in gathered_lengths for item in sublist]
        if gathered_embeddings and isinstance(gathered_embeddings[0], list):
            # Check if it's a list of numpy arrays or nested lists
            if len(gathered_embeddings) > 0 and isinstance(gathered_embeddings[0][0], np.ndarray):
                # It's a list of lists of numpy arrays, flatten one level
                gathered_embeddings = [item for sublist in gathered_embeddings for item in sublist]
            elif len(gathered_embeddings) > 0 and isinstance(gathered_embeddings[0], np.ndarray):
                # Already flat list of numpy arrays
                pass
        
        # Only save on main process
        if accelerator.is_local_main_process and gathered_embeddings:
            # Save embeddings using HDF5 format
            embeddings_path = os.path.join(output_dir, f"{split_name}_embeddings.h5")
            with h5py.File(embeddings_path, 'w') as f:
                # Create a dataset for unique_ids (as strings)
                unique_ids_encoded = [uid.encode('utf-8') for uid in gathered_unique_ids]
                f.create_dataset('unique_ids', data=unique_ids_encoded, compression='gzip')
                
                # Create a dataset for sequence lengths
                f.create_dataset('lengths', data=np.array(gathered_lengths, dtype=np.int32), compression='gzip')
                
                # Store embeddings in a ragged array format
                # We'll use a 2D array with max length and store actual lengths separately
                max_seq_len = max(gathered_lengths) if gathered_lengths else 0
                embed_dim = gathered_embeddings[0].shape[1] if gathered_embeddings else 5120
                
                # Create padded array
                padded_embeddings = np.zeros((len(gathered_embeddings), max_seq_len, embed_dim), dtype=np.float32)
                for idx, emb in enumerate(gathered_embeddings):
                    padded_embeddings[idx, :emb.shape[0]] = emb
                
                # Save with compression
                f.create_dataset('embeddings', data=padded_embeddings, compression='gzip', compression_opts=9)
                
                # Store metadata
                f.attrs['num_samples'] = len(gathered_embeddings)
                f.attrs['embed_dim'] = embed_dim
                f.attrs['max_seq_len'] = max_seq_len
            
            accelerator.print(f"💾 Saved {len(gathered_embeddings)} embeddings to {embeddings_path}")
            accelerator.print(f"   File size: {os.path.getsize(embeddings_path) / 1024 / 1024:.2f} MB")
    
    if accelerator.is_local_main_process:
        accelerator.print(f"\n✅ All embeddings generated and saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Pre-generate ESM embeddings for dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="embeddings",
        help="Directory to save embeddings",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for embedding generation",
    )
    args = parser.parse_args()
    
    # Load configuration
    cfg = load_config(args.config)
    data_config = cfg["dataset"]
    
    # Load tokenizer
    tokenizer = PTMTokenizer()
    
    # Load dataset
    dataset = get_ptm_dataset(
        tokenizer=tokenizer,
        dataset_location=data_config["dataset_location"],
        sequence_column_name=data_config["sequence_column_name"],
        val_size=data_config.get("val_size", 0),
        test_size=data_config.get("test_size", 0),
        split=data_config.get("split", True),
        subsample_size=data_config.get("subsample_size", None),
        split_seed=data_config.get("split_seed", None),
        max_sequence_length=data_config.get("max_sequence_length", None),
    )
    
    # Initialize accelerator for distributed processing
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    
    if accelerator.is_local_main_process:
        accelerator.print("🔄 Loading ESM model...")
    
    # Load ESM model
    esm_model, alphabet = esm.pretrained.esm2_t48_15B_UR50D()
    esm_model.eval()
    for param in esm_model.parameters():
        param.requires_grad = False
    batch_converter = alphabet.get_batch_converter()
    
    # Prepare ESM model with accelerator
    esm_model = accelerator.prepare(esm_model)
    
    if accelerator.is_local_main_process:
        accelerator.print(f"🚀 Using {accelerator.num_processes} GPU(s) for embedding generation")
    
    # Get max_sequence_length from training config (should match training setting)
    train_args = cfg.get("training", {})
    max_sequence_length = train_args.get("max_sequence_length", None)
    
    if accelerator.is_local_main_process:
        if max_sequence_length:
            accelerator.print(f"📏 Using max_sequence_length: {max_sequence_length} (sequences will be cropped)")
        else:
            accelerator.print(f"📏 No sequence length limit (using full sequences)")
    
    # Generate embeddings
    generate_embeddings_for_dataset(
        dataset=dataset,
        tokenizer=tokenizer,
        esm_model=esm_model,
        batch_converter=batch_converter,
        accelerator=accelerator,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_sequence_length=max_sequence_length,
    )


if __name__ == "__main__":
    main()


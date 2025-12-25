"""
Script to pre-generate ESM embeddings for all sequences in the dataset.
This allows faster training by loading pre-computed embeddings instead of computing them on-the-fly.
Supports multi-GPU generation using accelerate.
Supports both ESM2 15B and ESM3 7B models.
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
import fcntl
import time
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


def append_chunk_to_h5(chunk_data_by_ac_id, chunk_path, embed_dim, max_seq_len):
    """
    Append a chunk of embeddings to HDF5 file using new structure.
    
    New structure:
    - AC_ID (group name) -> {
        'embedding_seq': [seq1, seq2, seq3] (variable length list of embeddings)
        'range_seq': [[start1, end1], [start2, end2], [start3, end3]] (list of ranges)
    }
    
    @param {dict} chunk_data_by_ac_id - Dict mapping AC_ID to {
        'embeddings': [emb1, emb2, ...],  # List of numpy arrays
        'ranges': [(start1, end1), (start2, end2), ...]  # List of (start, end) tuples
    }
    @param {str} chunk_path - Path to the chunk file
    @param {int} embed_dim - Embedding dimension
    @param {int|None} max_seq_len - Maximum sequence length (None = auto-detect from data)
    """
    if not chunk_data_by_ac_id:
        return max_seq_len
    
    # Auto-detect max_seq_len from data if not provided
    if max_seq_len is None:
        max_len = 0
        for ac_id, data in chunk_data_by_ac_id.items():
            for emb in data['embeddings']:
                max_len = max(max_len, emb.shape[0])
        max_seq_len = max_len
    
    # Simple append with file lock for multi-process safety
    file_exists = os.path.exists(chunk_path)
    lock_path = chunk_path + '.lock'
    
    # Use file lock to ensure only one process writes at a time
    max_retries = 10
    retry_delay = 0.1  # 100ms
    lock_file = None
    
    for retry in range(max_retries):
        try:
            # Try to acquire lock
            lock_file = open(lock_path, 'w')
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            
            try:
                with h5py.File(chunk_path, 'a') as f:
                    # Set attributes if file is new
                    if not file_exists:
                        f.attrs['embed_dim'] = embed_dim
                        f.attrs['max_seq_len'] = max_seq_len
                    
                    # Write each AC_ID as a group
                    for ac_id, data in chunk_data_by_ac_id.items():
                        # Encode AC_ID for group name (h5py group names must be strings)
                        group_name = str(ac_id)
                        
                        if group_name in f:
                            # Group exists, append to existing data
                            group = f[group_name]
                            existing_embeddings = group['embedding_seq']
                            existing_ranges = group['range_seq']
                            
                            # Append new embeddings and ranges
                            num_existing = existing_embeddings.shape[0]
                            num_new = len(data['embeddings'])
                            new_size = num_existing + num_new
                            
                            # Resize and append embeddings
                            existing_embeddings.resize((new_size,))
                            for i, emb in enumerate(data['embeddings']):
                                # Pad embedding to max_seq_len
                                padded_emb = np.zeros((max_seq_len, embed_dim), dtype=np.float32)
                                padded_emb[:emb.shape[0], :] = emb
                                existing_embeddings[num_existing + i] = padded_emb
                            
                            # Resize and append ranges
                            existing_ranges.resize((new_size, 2))
                            for i, (start, end) in enumerate(data['ranges']):
                                existing_ranges[num_existing + i] = [start, end]
                        else:
                            # Create new group
                            group = f.create_group(group_name)
                            
                            # Create datasets for embeddings (variable length, padded to max_seq_len)
                            num_windows = len(data['embeddings'])
                            embedding_data = np.zeros((num_windows, max_seq_len, embed_dim), dtype=np.float32)
                            for i, emb in enumerate(data['embeddings']):
                                embedding_data[i, :emb.shape[0], :] = emb
                            
                            group.create_dataset('embedding_seq', data=embedding_data, 
                                                maxshape=(None, max_seq_len, embed_dim),
                                                dtype=np.float32, compression='gzip', compression_opts=9)
                            
                            # Create dataset for ranges
                            range_data = np.array(data['ranges'], dtype=np.int32)
                            group.create_dataset('range_seq', data=range_data, 
                                                maxshape=(None, 2), dtype=np.int32, compression='gzip')
                
                # Release lock
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                lock_file.close()
                break
            except Exception as e:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                lock_file.close()
                raise e
                
        except BlockingIOError:
            # Lock is held by another process, wait and retry
            if lock_file:
                lock_file.close()
            if retry < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise RuntimeError(f"Failed to acquire lock for {chunk_path} after {max_retries} retries")
    
    return max_seq_len


def merge_chunk_files(chunk_files, output_path, accelerator):
    """
    Merge multiple chunk HDF5 files into a single file (simple append mode).
    
    @param {list} chunk_files - List of chunk file paths to merge
    @param {str} output_path - Path to save merged file
    @param {Accelerator} accelerator - Accelerator instance for logging
    """
    if not chunk_files:
        return
    
    # Find global max_seq_len and embed_dim
    embed_dim = None
    global_max_seq_len = 0
    for chunk_file in chunk_files:
        if os.path.exists(chunk_file):
            with h5py.File(chunk_file, 'r') as f:
                global_max_seq_len = max(global_max_seq_len, f.attrs['max_seq_len'])
                if embed_dim is None:
                    embed_dim = f.attrs['embed_dim']
    
    # Simple append: read and append each chunk
    file_created = False
    for chunk_file in chunk_files:
        if not os.path.exists(chunk_file):
            continue
        
        with h5py.File(chunk_file, 'r') as f:
            unique_ids = [uid.decode('utf-8') for uid in f['unique_ids'][:]]
            lengths = f['lengths'][:]
            embeddings = f['embeddings'][:]
            
            # Pad if needed
            if embeddings.shape[1] < global_max_seq_len:
                padded = np.zeros((embeddings.shape[0], global_max_seq_len, embed_dim), dtype=np.float32)
                padded[:, :embeddings.shape[1], :] = embeddings
                embeddings = padded
            
            unique_ids_encoded = [uid.encode('utf-8') for uid in unique_ids]
            lengths_array = np.array(lengths, dtype=np.int32)
        
        # Append to output
        with h5py.File(output_path, 'a') as out_f:
            if not file_created:
                dt = h5py.special_dtype(vlen=bytes)
                out_f.create_dataset('unique_ids', data=unique_ids_encoded, maxshape=(None,), dtype=dt, compression='gzip')
                out_f.create_dataset('lengths', data=lengths_array, maxshape=(None,), dtype=np.int32, compression='gzip')
                out_f.create_dataset('embeddings', data=embeddings, maxshape=(None, global_max_seq_len, embed_dim),
                                   dtype=np.float32, compression='gzip', compression_opts=9)
                out_f.attrs['embed_dim'] = embed_dim
                out_f.attrs['max_seq_len'] = global_max_seq_len
                file_created = True
            else:
                current_size = out_f['unique_ids'].shape[0]
                new_size = current_size + len(unique_ids)
                
                out_f['unique_ids'].resize((new_size,))
                out_f['lengths'].resize((new_size,))
                out_f['embeddings'].resize((new_size, global_max_seq_len, embed_dim))
                
                out_f['unique_ids'][current_size:] = unique_ids_encoded
                out_f['lengths'][current_size:] = lengths_array
                out_f['embeddings'][current_size:] = embeddings
    
    # Clean up
    for chunk_file in chunk_files:
        try:
            os.remove(chunk_file)
        except:
            pass


def load_esm_model(model_name, accelerator):
    """
    Load ESM model (ESM2 15B or ESM3 7B) based on model name.
    
    @param model_name: Model name ('esm2_15b' or 'esm3_7b')
    @param accelerator: Accelerator instance for logging
    @return: Tuple of (esm_model, alphabet, batch_converter, repr_layer, model_type)
    """
    if accelerator.is_local_main_process:
        accelerator.print(f"🔄 Loading {model_name} model...")
    
    if model_name == "esm2_15b":
        # Load ESM2 15B model
        esm_model, alphabet = esm.pretrained.esm2_t48_15B_UR50D()
        repr_layer = 33  # ESM2 15B uses layer 33 for embeddings
        model_type = "esm2"
    elif model_name == "esm3_7b":
        # Load ESM3 7B model
        # Try different possible model identifiers for ESM3 7B
        try:
            # Try esm3-sm-open-v1 (7B model)
            esm_model, alphabet = esm.pretrained.load_model_and_alphabet("esm3-sm-open-v1")
            repr_layer = None  # Will use the last layer
            model_type = "esm3"
        except Exception as e1:
            try:
                # Try esm3-medium-2024-08 (alternative identifier)
                esm_model, alphabet = esm.pretrained.load_model_and_alphabet("esm3-medium-2024-08")
                repr_layer = None
                model_type = "esm3"
            except Exception as e2:
                if accelerator.is_local_main_process:
                    accelerator.print(f"❌ Failed to load ESM3 7B model")
                    accelerator.print(f"   Error 1 (esm3-sm-open-v1): {e1}")
                    accelerator.print(f"   Error 2 (esm3-medium-2024-08): {e2}")
                raise RuntimeError(f"Could not load ESM3 7B model. Please check if the model is available.")
    else:
        raise ValueError(f"Unknown model name: {model_name}. Supported: 'esm2_15b', 'esm3_7b'")
    
    esm_model.eval()
    for param in esm_model.parameters():
        param.requires_grad = False
    
    batch_converter = alphabet.get_batch_converter()
    
    # Determine the actual representation layer for ESM3
    if model_type == "esm3" and repr_layer is None:
        # ESM3 typically uses the last layer, get the number of layers
        try:
            # Try to get layer count from model structure
            if hasattr(esm_model, "num_layers"):
                repr_layer = esm_model.num_layers - 1
            elif hasattr(esm_model, "encoder") and hasattr(esm_model.encoder, "num_layers"):
                repr_layer = esm_model.encoder.num_layers - 1
            elif hasattr(esm_model, "layers"):
                repr_layer = len(esm_model.layers) - 1
            else:
                # Default: use -1 to indicate last layer (will be determined at runtime)
                repr_layer = -1
        except Exception:
            # If we can't determine, use -1 to indicate last layer
            repr_layer = -1
    
    if accelerator.is_local_main_process:
        accelerator.print(f"✅ Loaded {model_name} model")
        accelerator.print(f"   Using representation layer: {repr_layer}")
    
    return esm_model, alphabet, batch_converter, repr_layer, model_type


def extract_esm_embeddings(esm_model, batch_tokens, repr_layer, model_type, accelerator):
    """
    Extract embeddings from ESM model (supports both ESM2 and ESM3).
    
    @param esm_model: ESM model instance
    @param batch_tokens: Tokenized input tokens
    @param repr_layer: Layer number to extract embeddings from
    @param model_type: Model type ('esm2' or 'esm3')
    @param accelerator: Accelerator instance for error handling
    @return: Embeddings tensor
    """
    with torch.no_grad():
        if model_type == "esm2":
            # ESM2 API
            out = esm_model(batch_tokens, repr_layers=[repr_layer], return_contacts=False)
            embeddings = out["representations"][repr_layer]
        elif model_type == "esm3":
            # ESM3 API - try different approaches
            try:
                # Try ESM3 standard API
                out = esm_model(batch_tokens, repr_layers=[repr_layer] if repr_layer >= 0 else None, return_contacts=False)
                if isinstance(out, dict) and "representations" in out:
                    if repr_layer >= 0:
                        embeddings = out["representations"][repr_layer]
                    else:
                        # Use the last layer
                        last_layer = max(out["representations"].keys())
                        embeddings = out["representations"][last_layer]
                else:
                    # ESM3 might return embeddings directly
                    embeddings = out if isinstance(out, torch.Tensor) else out[0]
            except Exception as e:
                # Fallback: try to get last hidden state
                try:
                    out = esm_model(batch_tokens)
                    if isinstance(out, tuple):
                        embeddings = out[0]  # First element is usually hidden states
                    elif isinstance(out, dict):
                        # Try common keys
                        if "hidden_states" in out:
                            embeddings = out["hidden_states"][-1]  # Last layer
                        elif "representations" in out:
                            last_layer = max(out["representations"].keys())
                            embeddings = out["representations"][last_layer]
                        else:
                            raise RuntimeError(f"Unknown ESM3 output format: {out.keys()}")
                    else:
                        embeddings = out
                except Exception as e2:
                    if accelerator.is_local_main_process:
                        accelerator.print(f"❌ Error extracting ESM3 embeddings: {e}")
                        accelerator.print(f"   Fallback error: {e2}")
                    raise RuntimeError(f"Failed to extract embeddings from ESM3 model: {e}")
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    return embeddings


def generate_embeddings_for_dataset(
    dataset,
    tokenizer,
    esm_model,
    batch_converter,
    accelerator,
    output_dir,
    batch_size=8,
    max_sequence_length=None,
    chunk_size_gb=100,
    original_sequences_dict=None,
    repr_layer=33,
    model_type="esm2",
):
    """
    Generate ESM embeddings for all sequences in the dataset.
    Uses fixed-size chunking to avoid memory explosion.
    
    @param dataset: Dataset dict with train/val/test splits.
    @param tokenizer: Tokenizer instance.
    @param esm_model: ESM model (prepared by accelerator).
    @param batch_converter: ESM alphabet batch converter.
    @param accelerator: Accelerator instance for distributed processing.
    @param output_dir: Directory to save embeddings.
    @param batch_size: Batch size for embedding generation.
    @param max_sequence_length: Maximum sequence length (same as training). If None, no limit.
    @param chunk_size_gb: Target chunk size in GB (default: 100GB)
    @param original_sequences_dict: Optional dict mapping AC_ID to original sequence (without PTM). 
                                    If provided, will use original sequences for ESM input instead of PTM sequences.
    """
    device = accelerator.device
    
    # Create output directory
    if accelerator.is_local_main_process:
        os.makedirs(output_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    
    # Process each split
    for split_name in ["train", "val", "test"]:
        if split_name not in dataset or dataset[split_name] is None:
            continue
        
        split_dataset = dataset[split_name]
        if accelerator.is_local_main_process:
            accelerator.print(f"\n📊 Processing {split_name} set ({len(split_dataset)} samples)...")
        
        # All processes write to the same final file (append mode)
        embeddings_path = os.path.join(output_dir, f"{split_name}_embeddings.h5")
        
        # 🔧 Resume functionality: Load already processed AC_IDs
        processed_unique_ids = set()
        if os.path.exists(embeddings_path):
            try:
                with h5py.File(embeddings_path, 'r') as f:
                    # New structure: each AC_ID is a group
                    for ac_id in f.keys():
                        if isinstance(f[ac_id], h5py.Group):
                            processed_unique_ids.add(ac_id)
                    
                    if accelerator.is_local_main_process:
                        accelerator.print(f"📋 Found existing embeddings file with {len(processed_unique_ids)} AC_IDs")
                        accelerator.print(f"🔄 Resuming from checkpoint: {len(processed_unique_ids)} AC_IDs already processed")
            except Exception as e:
                if accelerator.is_local_main_process:
                    accelerator.print(f"⚠️  Warning: Could not read existing file for resume: {e}")
        
        # Distribute samples across processes
        num_processes = accelerator.num_processes
        process_rank = accelerator.process_index
        
        # Calculate indices for this process
        total_samples = len(split_dataset)
        samples_per_process = total_samples // num_processes
        start_idx = process_rank * samples_per_process
        end_idx = start_idx + samples_per_process if process_rank < num_processes - 1 else total_samples
        
        # Estimate chunk size in samples
        estimated_bytes_per_sample = 2 * 1024 * 1024  # 2MB per sample (compressed estimate)
        chunk_size_samples = int(chunk_size_gb * 1024 * 1024 * 1024 / estimated_bytes_per_sample)
        chunk_size_samples = max(1000, min(chunk_size_samples, 50000))  # Between 1K and 50K samples
        
        if accelerator.is_local_main_process:
            accelerator.print(f"📦 Using chunk size: ~{chunk_size_samples} samples per chunk (~{chunk_size_gb}GB)")
            accelerator.print(f"💾 All processes writing directly to: {embeddings_path}")
        
        # Process in batches and save in chunks
        process_indices = range(start_idx, end_idx)
        if accelerator.is_local_main_process:
            pbar = tqdm(range(0, len(process_indices), batch_size), desc=f"Generating {split_name} embeddings")
        else:
            pbar = range(0, len(process_indices), batch_size)
        
        # Per-process chunk storage - organized by AC_ID
        # Structure: {AC_ID: {'embeddings': [emb1, emb2, ...], 'ranges': [(start1, end1), ...]}}
        chunk_data_by_ac_id = {}
        chunk_idx = 0
        embed_dim = None
        max_seq_len = max_sequence_length  # Will be determined on first save if None
        
        # 🔧 Resume statistics
        skipped_sequences = 0
        processed_sequences = 0
        
        for batch_offset in pbar:
            batch_samples = []
            batch_indices = []
            
            batch_start = start_idx + batch_offset
            batch_end = min(batch_start + batch_size, end_idx)
            
            for j in range(batch_start, batch_end):
                sample = split_dataset[j]
                batch_samples.append(sample)
                batch_indices.append(j)
            
            # Get sequences and unique_ids (AC_IDs)
            ac_ids = [s.get("unique_id", f"{split_name}_{j}") for j, s in zip(batch_indices, batch_samples)]
            
            # 🔄 Use original sequences if provided, otherwise use PTM sequences
            use_original_seq = original_sequences_dict is not None
            if use_original_seq:
                # Use original sequences (without PTM) for ESM input
                sequences_for_esm = [original_sequences_dict.get(ac_id, s["sequence"]) 
                                    for ac_id, s in zip(ac_ids, batch_samples)]
                if accelerator.is_local_main_process and batch_offset == 0:
                    accelerator.print(f"📝 Using original sequences (without PTM) for ESM input")
            else:
                # Use PTM sequences (will mask PTM tokens later)
                sequences_for_esm = [s["sequence"] for s in batch_samples]
            
            # Process each sequence with window extraction strategy
            for idx, ac_id in enumerate(ac_ids):
                # 🔧 Resume check: Skip if this AC_ID is already processed
                if ac_id in processed_unique_ids:
                    skipped_sequences += 1
                    continue
                
                # Get the sequence for ESM
                seq_for_esm = sequences_for_esm[idx]
                
                if use_original_seq:
                    # For original sequences: work directly with string length
                    # Original sequences only contain standard amino acids, so length = len(seq)
                    seq_len = len(seq_for_esm)
                    # Compute windows based on sequence string length
                    windows = compute_extraction_windows(seq_len, max_sequence_length)
                else:
                    # For PTM sequences: tokenize first to get correct length
                    tokenized = tokenizer(
                        [seq_for_esm],
                        add_special_tokens=True,
                        max_sequence_length=None,
                    )
                    input_ids = torch.tensor(tokenized[0], device=device)
                    seq_len = len(input_ids)
                    # Compute windows based on tokenized length
                    windows = compute_extraction_windows(seq_len, max_sequence_length)
                
                # Collect all windows for this AC_ID
                embeddings_for_ac_id = []
                ranges_for_ac_id = []
                
                # Process each window separately
                for win_idx, (window_start, window_end) in enumerate(windows):
                    if use_original_seq:
                        # Extract window directly from original sequence string
                        window_seq = seq_for_esm[window_start:window_end]
                        window_len = len(window_seq)
                        
                        # Prepare ESM input directly from original sequence string
                        esm_input = (0, window_seq)
                    else:
                        # Extract window from tokenized sequence
                        window_ids = input_ids[window_start:window_end]
                        window_len = window_end - window_start
                        
                        # Prepare ESM input: replace PTM tokens with mask
                        window_ids = window_ids.unsqueeze(0)
                        esm_input_ids = make_esm_input_ids(window_ids, tokenizer)
                        decoded_str = tokenizer.decode(esm_input_ids[0].detach().cpu().tolist())
                        # Remove <cls> and <eos> from decoded string before passing to ESM
                        # ESM's batch_converter will add its own <cls> and <eos>
                        decoded_str_clean = decoded_str.replace("<cls>", "").replace("<eos>", "")
                        esm_input = (0, decoded_str_clean)
                    
                    # Compute ESM embedding
                    batch_labels, batch_strs, batch_tokens = batch_converter([esm_input])
                    
                    # Handle token format differences between ESM2 and ESM3
                    if model_type == "esm2":
                        # ESM2: remove <cls> and <eos> tokens (first and last)
                        batch_tokens = batch_tokens[..., 1:-1].to(device)
                    elif model_type == "esm3":
                        # ESM3: may have different token format, adjust if needed
                        batch_tokens = batch_tokens.to(device)
                        # ESM3 might include special tokens differently, check and adjust
                        if batch_tokens.shape[-1] > window_len + 2:
                            # Remove special tokens if present
                            batch_tokens = batch_tokens[..., 1:-1]
                    
                    # Extract embeddings using unified function
                    window_emb_all = extract_esm_embeddings(
                        esm_model, batch_tokens, repr_layer, model_type, accelerator
                    )
                    window_emb = window_emb_all[0]  # Get first (and only) sequence in batch
                    
                    # Extract actual window embedding
                    window_emb_actual = window_emb[:window_len].cpu().numpy()
                    
                    if embed_dim is None:
                        embed_dim = window_emb_actual.shape[1]
                    
                    # Store embedding and range for this window
                    embeddings_for_ac_id.append(window_emb_actual)
                    ranges_for_ac_id.append((window_start, window_end))
                
                # Store all windows for this AC_ID
                chunk_data_by_ac_id[ac_id] = {
                    'embeddings': embeddings_for_ac_id,
                    'ranges': ranges_for_ac_id
                }
                processed_sequences += 1
                
                # Save chunk when it reaches target size (by number of AC_IDs)
                if len(chunk_data_by_ac_id) >= chunk_size_samples:
                    max_seq_len = append_chunk_to_h5(
                        chunk_data_by_ac_id, embeddings_path, embed_dim, max_seq_len
                    )
                    if accelerator.is_local_main_process:
                        accelerator.print(f"💾 Appended chunk {chunk_idx} ({len(chunk_data_by_ac_id)} AC_IDs)")
                    chunk_data_by_ac_id = {}
                    chunk_idx += 1
        
        # Save remaining data as last chunk (direct append to final file)
        if chunk_data_by_ac_id:
            append_chunk_to_h5(
                chunk_data_by_ac_id, embeddings_path, embed_dim, max_seq_len
            )
            if accelerator.is_local_main_process:
                accelerator.print(f"💾 Appended final chunk {chunk_idx} ({len(chunk_data_by_ac_id)} AC_IDs)")
        
        # Wait for all processes to finish
        accelerator.wait_for_everyone()
        
        # Print final file info
        if accelerator.is_local_main_process and os.path.exists(embeddings_path):
            file_size_mb = os.path.getsize(embeddings_path) / 1024 / 1024
            accelerator.print(f"✅ Saved {split_name} embeddings to {embeddings_path}")
            accelerator.print(f"   File size: {file_size_mb:.2f} MB ({file_size_mb/1024:.2f} GB)")
            if skipped_sequences > 0:
                accelerator.print(f"   🔄 Resume: Skipped {skipped_sequences} already processed sequences")
                accelerator.print(f"   ✨ Processed {processed_sequences} new sequences")
    
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
    parser.add_argument(
        "--chunk_size_gb",
        type=float,
        default=100,
        help="Target chunk size in GB for saving embeddings (default: 100GB)",
    )
    parser.add_argument(
        "--use_original_sequence",
        action="store_true",
        help="Use original sequence (ori_seq) instead of PTM sequence for ESM input. "
             "Requires 'ori_seq' column in the dataset CSV file.",
    )
    parser.add_argument(
        "--original_sequence_column",
        type=str,
        default="ori_seq",
        help="Column name for original sequences in CSV (default: 'ori_seq')",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="esm2_15b",
        choices=["esm2_15b", "esm3_7b"],
        help="ESM model to use: 'esm2_15b' (ESM2 15B) or 'esm3_7b' (ESM3 7B). Default: 'esm2_15b'",
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
    
    # 🔄 Load original sequences if requested
    original_sequences_dict = None
    if args.use_original_sequence:
        import pandas as pd
        if accelerator.is_local_main_process:
            accelerator.print(f"📖 Loading original sequences from column '{args.original_sequence_column}'...")
        
        df = pd.read_csv(data_config["dataset_location"])
        df.drop(df.filter(regex="^Unnamed"), axis=1, inplace=True)
        
        if args.original_sequence_column not in df.columns:
            raise ValueError(
                f"❌ Column '{args.original_sequence_column}' not found in dataset. "
                f"Available columns: {list(df.columns)}"
            )
        
        # Get AC_ID column
        if "AC_ID" in df.columns:
            ac_ids = df["AC_ID"].tolist()
        else:
            ac_ids = df.index.tolist()
        
        # Create mapping: AC_ID -> original_sequence
        original_sequences_dict = {
            str(ac_id): str(ori_seq) 
            for ac_id, ori_seq in zip(ac_ids, df[args.original_sequence_column].tolist())
        }
        
        if accelerator.is_local_main_process:
            accelerator.print(f"✅ Loaded {len(original_sequences_dict)} original sequences")
    
    # Load ESM model based on user choice
    esm_model, alphabet, batch_converter, repr_layer, model_type = load_esm_model(
        args.model, accelerator
    )
    
    # Prepare ESM model with accelerator
    esm_model = accelerator.prepare(esm_model)
    
    if accelerator.is_local_main_process:
        accelerator.print(f"🚀 Using {accelerator.num_processes} GPU(s) for embedding generation")
        accelerator.print(f"📦 Model type: {model_type}, Representation layer: {repr_layer}")
    
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
        chunk_size_gb=args.chunk_size_gb,
        original_sequences_dict=original_sequences_dict,
        repr_layer=repr_layer,
        model_type=model_type,
    )


if __name__ == "__main__":
    main()


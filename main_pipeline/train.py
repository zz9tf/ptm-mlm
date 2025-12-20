import torch
import yaml
import argparse
from transformers.trainer import DataLoader
import os
import esm
from accelerate import Accelerator, DistributedType
from accelerate.utils import DistributedDataParallelKwargs
from tqdm import tqdm
from datetime import datetime

from getters.ptm_dataset import (
    DataCollatorWithPadding,
    SequenceLengthSampler,
)
from accelerate.utils import set_seed
from getters.ptm_dataset import get_ptm_dataset
from getters.tokenizer import PTMTokenizer
from utils.log import TrainLogger
from utils.mask import Masker
from utils.scheduler import Esm2LRScheduler
from utils.config import load_config
from utils.distributed import is_main_process_from_env
from utils.checkpoint import load_ckpt
from utils.esm_utils import make_esm_input_ids, compute_esm_embedding
from utils.loss import mlm_loss
from utils.embedding_loader import EmbeddingLoader
from models.mamba.lm import MambaLMHeadModel

def train(
    config_dict: dict,
    model_config: dict,
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    tokenizer,
    masker: Masker,
    logger,
    accelerator: Accelerator,
    checkpoint_dir: str,
    test_loader=None,
):
    """
    Training function.
    @param config_dict: Configuration dictionary.
    @param model_config: Model configuration dictionary.
    @param model: Model instance.
    @param train_loader: Training data loader.
    @param val_loader: Validation data loader.
    @param optimizer: Optimizer instance.
    @param scheduler: Learning rate scheduler instance.
    @param tokenizer: Tokenizer instance.
    @param masker: Masker instance.
    @param logger: Logger instance.
    @param accelerator: Accelerator instance.
    @param checkpoint_dir: Checkpoint directory with timestamp.
    @param test_loader: Test data loader (optional).
    """
    train_args = config_dict["training"]
    save_dir = checkpoint_dir
    device = accelerator.device
    # Only create directory and set paths on main process
    if accelerator.is_local_main_process:
        os.makedirs(save_dir, exist_ok=True)
        best_ckpt_path = os.path.join(save_dir, "best.ckpt")
        last_ckpt_path = os.path.join(save_dir, "last.ckpt")
    else:
        # Non-main processes don't need checkpoint paths
        best_ckpt_path = None
        last_ckpt_path = None
    best_loss = float("inf")

    # Initialize ESM model or embedding loader
    use_precomputed_embeddings = train_args.get("use_precomputed_embeddings", False)
    embeddings_dir = train_args.get("embeddings_dir", None)
    embedding_loader = None
    
    # Get seed from config for reproducible window selection
    seed = config_dict.get("seed", None)
    
    if train_args.get("use_esm", False):
        if use_precomputed_embeddings and embeddings_dir:
            # Load pre-computed embeddings
            if accelerator.is_local_main_process:
                print(f"📦 Loading pre-computed embeddings from {embeddings_dir}")
                if seed is not None:
                    print(f"🔑 Using seed {seed} for reproducible window selection")
            embedding_loader = EmbeddingLoader(embeddings_dir, seed=seed)
            esm_model = None
            batch_converter = None
        else:
            # Use ESM model for on-the-fly computation
            esm_model, alphabet = esm.pretrained.esm2_t48_15B_UR50D()
            batch_converter = alphabet.get_batch_converter()
            esm_model.eval()
            for param in esm_model.parameters():
                param.requires_grad = False
            # Prepare ESM model with accelerator for distributed training support
            esm_model = accelerator.prepare(esm_model)
    else:
        esm_model = None
        batch_converter = None
    
    model_to_save = model if accelerator.distributed_type==DistributedType.NO else model.module
    masking_fn = masker.random_or_random_and_ptm_mask
    total_steps = 0
    
    # Calculate total steps for progress bar
    total_train_steps = len(train_loader) * train_args.get("num_train_epochs", 10)
    
    for epoch in range(train_args.get("num_train_epochs", 10)):
        # Create progress bar for training (only on main process)
        train_pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{train_args.get('num_train_epochs', 10)}",
            disable=not accelerator.is_local_main_process,
            leave=True,
        )
        
        for batch in train_pbar:
            optimizer.zero_grad()
            input_ids = batch["input_ids"]
            unique_ids = batch.get("unique_ids", None)
            
            # masking
            masked_input_ids, pred_mask = masking_fn(input_ids)
            
            # compute ESM embedding
            if train_args.get("use_esm", False):
                if embedding_loader is not None and unique_ids is not None:
                    # Use pre-computed embeddings (random window selection for training)
                    embedding = embedding_loader.get_embeddings(unique_ids, device, is_training=True)
                    if embedding is None:
                        # Fallback to on-the-fly computation if embeddings not found
                        esm_input_ids = make_esm_input_ids(input_ids, tokenizer)
                        masked_esm_input_ids = esm_input_ids.clone()
                        masked_esm_input_ids[pred_mask] = tokenizer.mask_token_id
                        embedding = compute_esm_embedding(
                            tokenizer, esm_model, batch_converter, masked_esm_input_ids, accelerator
                        )
                    else:
                        # Align embedding dimensions with input_ids (handle padding)
                        if embedding.shape[1] < input_ids.shape[1]:
                            # Pad embedding
                            pad_size = input_ids.shape[1] - embedding.shape[1]
                            embedding = torch.nn.functional.pad(embedding, (0, 0, 0, pad_size))
                        elif embedding.shape[1] > input_ids.shape[1]:
                            # Crop embedding
                            embedding = embedding[:, :input_ids.shape[1]]
                else:
                    # On-the-fly computation
                    esm_input_ids = make_esm_input_ids(input_ids, tokenizer)
                    masked_esm_input_ids = esm_input_ids.clone()
                    masked_esm_input_ids[pred_mask] = tokenizer.mask_token_id
                    embedding = compute_esm_embedding(
                        tokenizer, esm_model, batch_converter, masked_esm_input_ids, accelerator
                    )
            else:
                embedding = None
            
            # forward pass
            outputs = model(masked_input_ids, embedding=embedding)
            logits = outputs.logits
            loss = mlm_loss(logits, input_ids, pred_mask)
            
            # backward pass
            accelerator.backward(loss)
            
            # compute accuracy
            preplexity = torch.exp(loss)
            acc = (logits.argmax(dim=-1) == input_ids)[pred_mask].float().mean()
            # Compute PTM accuracy - handle case when no PTM tokens are masked
            ptm_mask = pred_mask & tokenizer.is_ptm_token(input_ids).to(device)
            if ptm_mask.any():
                ptm_acc = (
                    (logits.argmax(dim=-1) == input_ids)[ptm_mask]
                    .float()
                    .mean()
                )
            else:
                # No PTM tokens masked in this batch, set accuracy to 0.0
                ptm_acc = torch.tensor(0.0, device=device)
            
            # update parameters
            optimizer.step()
            # update learning rate
            scheduler.step()
            
            # Update progress bar with metrics (only on main process)
            # Get current learning rate from optimizer (after scheduler.step())
            current_lr = optimizer.param_groups[0]['lr'] if accelerator.is_local_main_process else 0.0
            if accelerator.is_local_main_process:
                train_pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{acc.item():.4f}",
                    "ptm_acc": f"{ptm_acc.item():.4f}",
                    "ppl": f"{preplexity.item():.2f}",
                    "lr": f"{current_lr:.2e}",
                })
                logger.log(
                    {
                        "train_loss": loss.item(),
                        "train_preplexity": preplexity.item(),
                        "train_acc": acc.item(),
                        "train_ptm_acc": ptm_acc.item(),
                        "act_bs": input_ids.shape[0],
                        "act_seq_len": input_ids.shape[1],
                    }
                )
            total_steps += 1
            if total_steps % train_args.get("log_steps", 100) == 0:
                # evaluate on validation set
                model.eval()
                # Create progress bar for validation (only on main process)
                val_pbar = tqdm(
                    val_loader,
                    desc="Validating",
                    disable=not accelerator.is_local_main_process,
                    leave=False,
                )
                # Accumulate validation metrics across all batches
                val_losses = []
                val_ppls = []
                val_accs = []
                val_ptm_accs = []
                
                for val_batch in val_pbar:
                    with torch.no_grad():
                        input_ids = val_batch["input_ids"]
                        unique_ids = val_batch.get("unique_ids", None)
                        masked_input_ids, pred_mask = masking_fn(input_ids)
                        
                        if train_args.get("use_esm", False):
                            if embedding_loader is not None and unique_ids is not None:
                                # Use pre-computed embeddings (fixed window selection for validation)
                                embedding = embedding_loader.get_embeddings(unique_ids, device, is_training=False)
                                if embedding is None:
                                    esm_input_ids = make_esm_input_ids(input_ids, tokenizer)
                                    masked_esm_input_ids = esm_input_ids.clone()
                                    masked_esm_input_ids[pred_mask] = tokenizer.mask_token_id
                                    embedding = compute_esm_embedding(
                                        tokenizer, esm_model, batch_converter, masked_esm_input_ids, accelerator
                                    )
                                else:
                                    if embedding.shape[1] < input_ids.shape[1]:
                                        pad_size = input_ids.shape[1] - embedding.shape[1]
                                        embedding = torch.nn.functional.pad(embedding, (0, 0, 0, pad_size))
                                    elif embedding.shape[1] > input_ids.shape[1]:
                                        embedding = embedding[:, :input_ids.shape[1]]
                            else:
                                esm_input_ids = make_esm_input_ids(input_ids, tokenizer)
                                masked_esm_input_ids = esm_input_ids.clone()
                                masked_esm_input_ids[pred_mask] = tokenizer.mask_token_id
                                embedding = compute_esm_embedding(
                                    tokenizer, esm_model, batch_converter, masked_esm_input_ids, accelerator
                                )
                        else:
                            embedding = None
                        outputs = model(masked_input_ids, embedding=embedding)
                        logits = outputs.logits
                        loss = mlm_loss(logits, input_ids, pred_mask)
                    # compute accuracy
                    preplexity = torch.exp(loss)
                    acc = (logits.argmax(dim=-1) == input_ids)[pred_mask].float().mean()
                    # Compute PTM accuracy - handle case when no PTM tokens are masked
                    ptm_mask = pred_mask & tokenizer.is_ptm_token(input_ids).to(device)
                    if ptm_mask.any():
                        ptm_acc = (
                            (logits.argmax(dim=-1) == input_ids)[ptm_mask]
                            .float()
                            .mean()
                        )
                    else:
                        # No PTM tokens masked in this batch, set accuracy to 0.0
                        ptm_acc = torch.tensor(0.0, device=device)
                    
                    # Accumulate metrics for averaging
                    val_losses.append(loss.item())
                    val_ppls.append(preplexity.item())
                    val_accs.append(acc.item())
                    val_ptm_accs.append(ptm_acc.item())
                    
                    # Update validation progress bar (only on main process)
                    if accelerator.is_local_main_process:
                        val_pbar.set_postfix({
                            "val_loss": f"{loss.item():.4f}",
                            "val_acc": f"{acc.item():.4f}",
                            "val_ptm_acc": f"{ptm_acc.item():.4f}",
                        })
                        # Log each batch's metrics for tracking
                        logger.log(
                            {
                                "Epoch": epoch,
                                "Step": total_steps,
                                "val_loss": loss.item(),
                                "val_preplexity": preplexity.item(),
                                "val_acc": acc.item(),
                                "val_ptm_acc": ptm_acc.item(),
                            }
                        )
                
                # Calculate and log average validation metrics
                if accelerator.is_local_main_process and len(val_losses) > 0:
                    avg_val_loss = sum(val_losses) / len(val_losses)
                    avg_val_ppl = sum(val_ppls) / len(val_ppls)
                    avg_val_acc = sum(val_accs) / len(val_accs)
                    avg_val_ptm_acc = sum(val_ptm_accs) / len(val_ptm_accs)
                    
                    # Log average validation metrics
                    logger.log(
                        {
                            "Epoch": epoch,
                            "Step": total_steps,
                            "avg_val_loss": avg_val_loss,
                            "avg_val_preplexity": avg_val_ppl,
                            "avg_val_acc": avg_val_acc,
                            "avg_val_ptm_acc": avg_val_ptm_acc,
                        }
                    )
                    accelerator.print(f"📊 Avg Validation - Loss: {avg_val_loss:.4f}, Acc: {avg_val_acc:.4f}, PTM Acc: {avg_val_ptm_acc:.4f}, PPL: {avg_val_ppl:.2f}")
                    
                    # save best model based on average validation loss
                    if avg_val_loss < best_loss:
                        best_loss = avg_val_loss
                        if best_ckpt_path is not None:  # Only save on main process
                            torch.save(
                                {"model": model_to_save.state_dict(), "config": model_config},
                                best_ckpt_path,
                            )
                            accelerator.print(f"💾 Best model saved! (val_loss: {avg_val_loss:.4f})")
                if accelerator.is_local_main_process:
                    # save last model
                    torch.save(
                        {"model": model_to_save.state_dict(), "config": model_config},
                        last_ckpt_path,
                    )
                    accelerator.print(f"Epoch {epoch}, Step {total_steps} finished!")
    if accelerator.is_local_main_process:
        # save last model
        torch.save(
            {"model": model_to_save.state_dict(), "config": model_config},
            last_ckpt_path,
        )
        accelerator.print(f"Training completed!")
        
        # Evaluate on test set if available
        if test_loader is not None:
            accelerator.print("🧪 Evaluating on test set...")
            model.eval()
            
            test_losses = []
            test_ppls = []
            test_accs = []
            test_ptm_accs = []
            
            test_pbar = tqdm(
                test_loader,
                desc="Testing",
                disable=not accelerator.is_local_main_process,
                leave=False,
            )
            
            for test_batch in test_pbar:
                with torch.no_grad():
                    input_ids = test_batch["input_ids"]
                    unique_ids = test_batch.get("unique_ids", None)
                    masked_input_ids, pred_mask = masking_fn(input_ids)
                    
                    if train_args.get("use_esm", False):
                        if embedding_loader is not None and unique_ids is not None:
                            # Use pre-computed embeddings (fixed window selection for test)
                            embedding = embedding_loader.get_embeddings(unique_ids, device, is_training=False)
                            if embedding is None:
                                esm_input_ids = make_esm_input_ids(input_ids, tokenizer)
                                masked_esm_input_ids = esm_input_ids.clone()
                                masked_esm_input_ids[pred_mask] = tokenizer.mask_token_id
                                embedding = compute_esm_embedding(
                                    tokenizer, esm_model, batch_converter, masked_esm_input_ids, accelerator
                                )
                            else:
                                if embedding.shape[1] < input_ids.shape[1]:
                                    pad_size = input_ids.shape[1] - embedding.shape[1]
                                    embedding = torch.nn.functional.pad(embedding, (0, 0, 0, pad_size))
                                elif embedding.shape[1] > input_ids.shape[1]:
                                    embedding = embedding[:, :input_ids.shape[1]]
                        else:
                            esm_input_ids = make_esm_input_ids(input_ids, tokenizer)
                            masked_esm_input_ids = esm_input_ids.clone()
                            masked_esm_input_ids[pred_mask] = tokenizer.mask_token_id
                            embedding = compute_esm_embedding(
                                tokenizer, esm_model, batch_converter, masked_esm_input_ids, accelerator
                            )
                    else:
                        embedding = None
                    outputs = model(masked_input_ids, embedding=embedding)
                    logits = outputs.logits
                    loss = mlm_loss(logits, input_ids, pred_mask)
                
                # compute accuracy
                preplexity = torch.exp(loss)
                acc = (logits.argmax(dim=-1) == input_ids)[pred_mask].float().mean()
                # Compute PTM accuracy - handle case when no PTM tokens are masked
                ptm_mask = pred_mask & tokenizer.is_ptm_token(input_ids).to(device)
                if ptm_mask.any():
                    ptm_acc = (
                        (logits.argmax(dim=-1) == input_ids)[ptm_mask]
                        .float()
                        .mean()
                    )
                else:
                    # No PTM tokens masked in this batch, set accuracy to 0.0
                    ptm_acc = torch.tensor(0.0, device=device)
                
                # Accumulate metrics
                test_losses.append(loss.item())
                test_ppls.append(preplexity.item())
                test_accs.append(acc.item())
                test_ptm_accs.append(ptm_acc.item())
                
                # Update test progress bar
                if accelerator.is_local_main_process:
                    test_pbar.set_postfix({
                        "test_loss": f"{loss.item():.4f}",
                        "test_acc": f"{acc.item():.4f}",
                        "test_ptm_acc": f"{ptm_acc.item():.4f}",
                    })
                    # Log each batch's test metrics
                    logger.log(
                        {
                            "test_loss": loss.item(),
                            "test_preplexity": preplexity.item(),
                            "test_acc": acc.item(),
                            "test_ptm_acc": ptm_acc.item(),
                        }
                    )
            
            # Calculate and log average test metrics
            if accelerator.is_local_main_process and len(test_losses) > 0:
                avg_test_loss = sum(test_losses) / len(test_losses)
                avg_test_ppl = sum(test_ppls) / len(test_ppls)
                avg_test_acc = sum(test_accs) / len(test_accs)
                avg_test_ptm_acc = sum(test_ptm_accs) / len(test_ptm_accs)
                
                # Log average test metrics
                logger.log(
                    {
                        "avg_test_loss": avg_test_loss,
                        "avg_test_preplexity": avg_test_ppl,
                        "avg_test_acc": avg_test_acc,
                        "avg_test_ptm_acc": avg_test_ptm_acc,
                    }
                )
                accelerator.print(f"📊 Avg Test - Loss: {avg_test_loss:.4f}, Acc: {avg_test_acc:.4f}, PTM Acc: {avg_test_ptm_acc:.4f}, PPL: {avg_test_ppl:.2f}")
        
        # Finalize logger to write all metrics to disk
        if hasattr(logger, 'finalize'):
            logger.finalize()
    accelerator.print(f"Training completed!")
    
def main():
    """
    Main training function.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="PTM-Mamba Training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to configuration YAML file",
    )
    args = parser.parse_args()
    
    # Load configuration
    cfg = load_config(args.config)
    
    # Access configuration groups
    data_config = cfg["dataset"]
    train_args = cfg["training"]
    model_config = cfg["model"]
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    # Now Accelerator initializes and can only see GPUs specified in CUDA_VISIBLE_DEVICES
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    device = accelerator.device
    set_seed(cfg.get("seed", 42))
    
    # Print GPU usage summary
    if torch.cuda.is_available() and accelerator.is_local_main_process:
        num_processes = accelerator.num_processes
        print(f"GPUs in use: {list(range(num_processes))}")
    
    # Create output directory with timestamp (only on main process)
    # All outputs (config, checkpoints, metrics) will be saved here
    exp_name = cfg.get("exp_name", "ptm-mamba")
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_dir = f"outputs/{exp_name}-{timestamp}"
    
    if accelerator.is_local_main_process:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        # Save config to output directory
        config_output_path = os.path.join(output_dir, "config.yaml")
        with open(config_output_path, 'w', encoding='utf-8') as f:
            yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
        print(f"📝 Config saved to: {config_output_path}")
        print(f"💾 Output directory: {output_dir}")
    else:
        output_dir = None
    
    if "wandb" in cfg.get("report_to", []):
        import wandb

        if accelerator.is_local_main_process:
            wandb.init(
                project="PTM-Mamba", config=dict(cfg), name=output_dir
            )
        logger = wandb
    else:
        # Initialize logger with output_dir and config
        logger_save_dir = output_dir if accelerator.is_local_main_process else None
        logger = TrainLogger(save_dir=logger_save_dir, config=cfg if accelerator.is_local_main_process else None)

    tokenizer = PTMTokenizer()
    # Use main seed for split_seed if not explicitly provided (for reproducibility)
    split_seed = data_config.get("split_seed", cfg.get("seed", 42))
    dataset = get_ptm_dataset(
        tokenizer=tokenizer,
        dataset_location=data_config["dataset_location"],
        sequence_column_name=data_config["sequence_column_name"],
        val_size=data_config.get("val_size", 0),
        test_size=data_config.get("test_size", 0),
        split=data_config.get("split", True),
        subsample_size=data_config.get("subsample_size", None),
        split_seed=split_seed,
        max_sequence_length=data_config.get("max_sequence_length", None),
    )
    
    # Log dataset split information and save split mapping (only on main process)
    if accelerator.is_local_main_process:
        dataset_info = {
            "train_size": len(dataset["train"]) if "train" in dataset and dataset["train"] is not None else 0,
            "val_size": len(dataset["val"]) if "val" in dataset and dataset["val"] is not None else 0,
            "test_size": len(dataset["test"]) if "test" in dataset and dataset["test"] is not None else 0,
            "split_seed": data_config.get("split_seed", None),
            "val_size_config": data_config.get("val_size", 0),
            "test_size_config": data_config.get("test_size", 0),
        }
        logger.log({"dataset_split": dataset_info})
        accelerator.print(f"📊 Dataset split - Train: {dataset_info['train_size']}, Val: {dataset_info['val_size']}, Test: {dataset_info['test_size']}")
        
        # Save split mapping (unique_id -> split) to file for tracking
        import json
        split_mapping = dataset.get("split_mapping", {})
        
        # Also create mapping from samples if split_mapping not available
        if not split_mapping:
            split_mapping = {}
            for split_name in ["train", "val", "test"]:
                if split_name in dataset and dataset[split_name] is not None:
                    for sample in dataset[split_name].samples:
                        unique_id = sample.get("unique_id", "unknown")
                        split_mapping[unique_id] = split_name
        
        # Save split mapping to output directory
        if split_mapping:
            split_mapping_path = os.path.join(output_dir, "split_mapping.json")
            with open(split_mapping_path, 'w', encoding='utf-8') as f:
                json.dump(split_mapping, f, indent=2, ensure_ascii=False)
            accelerator.print(f"📝 Split mapping saved to: {split_mapping_path}")
            accelerator.print(f"   Total samples tracked: {len(split_mapping)}")
            # Log split mapping summary
            split_summary = {}
            for split_name in ["train", "val", "test"]:
                count = sum(1 for v in split_mapping.values() if v == split_name)
                split_summary[split_name] = count
            accelerator.print(f"   Split summary: {split_summary}")

    if train_args.get("resume_from_checkpoint"):
        model = load_ckpt(train_args["resume_from_checkpoint"], tokenizer, accelerator)
        accelerator.print(f"Model loaded from {train_args['resume_from_checkpoint']}")
    else:
        # Create a simple namespace-like object for model config
        from types import SimpleNamespace
        model_config_obj = SimpleNamespace(**model_config)
        model_config_obj.vocab_size = tokenizer.get_vocab_size()
        model = MambaLMHeadModel(config=model_config_obj, device=device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    accelerator.print(f"Number of parameters: {num_params:,}")
    sampler = SequenceLengthSampler(
        dataset["train"], train_args.get("sort_by_seq", True), train_args.get("sample_len_ascending", True))
    # Determine crop mode: use fixed crop if using pre-computed embeddings, random crop otherwise
    use_precomputed = train_args.get("use_precomputed_embeddings", False)
    random_crop_train = not use_precomputed  # Random crop for training (data augmentation) if not using pre-computed embeddings
    random_crop_eval = False  # Always use fixed crop for validation/test (consistency)
    
    train_loader = DataLoader(
        dataset["train"],
        batch_size=train_args["per_device_train_batch_size"],
        sampler=sampler,
        collate_fn=DataCollatorWithPadding(
            max_tokens=train_args["max_tokens_per_batch"],
            tokenizer=tokenizer,
            batch_by_tokens=True,
            max_seq_len=train_args.get("max_sequence_length", None),
            random_crop=random_crop_train,
        ),
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        dataset["val"],
        batch_size=train_args["per_device_train_batch_size"] // 2,
        collate_fn=DataCollatorWithPadding(
            max_tokens=train_args["max_tokens_per_batch"],
            tokenizer=tokenizer,
            batch_by_tokens=False,
            max_seq_len=train_args.get("max_sequence_length", None),
            random_crop=random_crop_eval,
        ),
        num_workers=0,
    )
    
    # Create test loader if test dataset exists
    test_loader = None
    if "test" in dataset and len(dataset["test"]) > 0:
        test_loader = DataLoader(
            dataset["test"],
            batch_size=train_args["per_device_train_batch_size"] // 2,
            collate_fn=DataCollatorWithPadding(
                max_tokens=train_args["max_tokens_per_batch"],
                tokenizer=tokenizer,
                batch_by_tokens=False,
                max_seq_len=train_args.get("max_sequence_length", None),
                random_crop=random_crop_eval,
            ),
            num_workers=0,
        )
    # Ensure lr is a float (YAML may read scientific notation as string)
    lr = float(train_args["lr"]) if isinstance(train_args["lr"], str) else train_args["lr"]
    
    optimizer = torch.optim.AdamW(
        model.parameters(), lr, betas=(0.9, 0.98), weight_decay=0.01
    )

    scheduler = Esm2LRScheduler(
        optimizer, last_epoch=-1, init_lr=lr, on_use=False
    )

    masker = Masker(tokenizer)
    # Prepare test_loader if it exists
    if test_loader is not None:
        model, optimizer, scheduler, train_loader, val_loader, test_loader = accelerator.prepare(
            model, optimizer, scheduler, train_loader, val_loader, test_loader
        )
    else:
        model, optimizer, scheduler, train_loader, val_loader = accelerator.prepare(
            model, optimizer, scheduler, train_loader, val_loader
        )

    train(
        config_dict=cfg,
        model_config=model_config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        tokenizer=tokenizer,
        masker=masker,
        logger=logger,
        accelerator=accelerator,
        checkpoint_dir=output_dir,
        test_loader=test_loader,
    )

      
if __name__ == "__main__":
    main()


import torch
import torch.nn.functional as F
import yaml
import argparse
from transformers.trainer import DataLoader
# Removed autocast and GradScaler - using pure FP16 training
import os
import json
import gc
from accelerate import Accelerator, DistributedType
from accelerate.utils import DistributedDataParallelKwargs
from tqdm import tqdm
from datetime import datetime

from getters.ptm_dataset_memmap import (
    get_ptm_dataset_memmap,
)
from accelerate.utils import set_seed
from getters.tokenizer import PTMTokenizer
from utils.log import TrainLogger
from utils.scheduler import Esm2LRScheduler
from utils.config import load_config
from utils.checkpoint import load_ckpt_from_output_dir
from utils.esm_utils import get_esm_embed_dim
from models.model import PTMModel


def get_last_training_metrics(metrics_path):
    """
    Get the last training metrics from metrics.json file, including epoch and best score.
    @param metrics_path: Path to metrics.json file.
    @returns: Dictionary with last training metrics, epoch, and best_val_loss, or None if not found.
    """
    if not os.path.exists(metrics_path):
        return None
    
    try:
        with open(metrics_path, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
        
        if not metrics:
            return None
        
        # Find the last training metrics (look for train_loss, avg_val_loss, etc.)
        last_train_metrics = {}
        last_val_metrics = {}
        last_test_metrics = {}
        
        # Find the last epoch and best validation loss
        last_epoch = None
        best_val_loss = None
        
        for entry in reversed(metrics):
            # Get last epoch (from entries with "Epoch" field)
            if "Epoch" in entry and last_epoch is None:
                last_epoch = entry.get("Epoch")
            
            # Get last training metrics
            if "train_loss" in entry and not last_train_metrics:
                last_train_metrics = {
                    "train_loss": entry.get("train_loss"),
                    "train_acc": entry.get("train_acc"),
                    "train_ptm_acc": entry.get("train_ptm_acc"),
                    "train_preplexity": entry.get("train_preplexity"),
                }
            
            # Get last validation metrics
            if "avg_val_loss" in entry and not last_val_metrics:
                last_val_metrics = {
                    "val_loss": entry.get("avg_val_loss"),
                    "val_acc": entry.get("avg_val_acc"),
                    "val_ptm_acc": entry.get("avg_val_ptm_acc"),
                    "val_preplexity": entry.get("avg_val_preplexity"),
                }
            
            # Track best validation loss (minimum across all validation entries)
            if "avg_val_loss" in entry:
                val_loss = entry.get("avg_val_loss")
                if val_loss is not None and (best_val_loss is None or val_loss < best_val_loss):
                    best_val_loss = val_loss
            
            # Get last test metrics if available
            if "avg_test_loss" in entry and not last_test_metrics:
                last_test_metrics = {
                    "test_loss": entry.get("avg_test_loss"),
                    "test_acc": entry.get("test_acc"),
                    "test_ptm_acc": entry.get("test_ptm_acc"),
                    "test_preplexity": entry.get("test_preplexity"),
                }
        
        result = {
            "train": last_train_metrics if last_train_metrics else None,
            "val": last_val_metrics if last_val_metrics else None,
            "test": last_test_metrics if last_test_metrics else None,
        }
        if last_epoch is not None:
            result["last_epoch"] = last_epoch
        if best_val_loss is not None:
            result["best_val_loss"] = best_val_loss
        return result
    except (json.JSONDecodeError, IOError) as e:
        print(f"⚠️  Warning: Could not read metrics file: {e}")
        return None


def train(
    config_dict: dict,
    model_config: dict,
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    tokenizer,
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
    # Initialize training state - load from metrics.json if resuming
    metrics_path = os.path.join(checkpoint_dir, "metrics.json")
    last_metrics = get_last_training_metrics(metrics_path) if os.path.exists(metrics_path) else None
    
    if last_metrics is not None:
        # Restore training progress from metrics.json
        # Note: last_epoch is the last completed epoch, so we should start from epoch + 1
        last_epoch = last_metrics.get("last_epoch", None)
        if last_epoch is not None:
            start_epoch = last_epoch + 1  # Start from the next epoch after the saved one
        else:
            start_epoch = 0  # No epoch found, start from beginning
        
        # Get best validation loss from metrics
        best_loss = last_metrics.get("best_val_loss", float("inf"))
        start_step = 0  # Step is not tracked in metrics, start from 0
        
        if accelerator.is_local_main_process:
            accelerator.print("=" * 80)
            accelerator.print("🔄 RESUMING TRAINING")
            accelerator.print("=" * 80)
            accelerator.print(f"📂 Checkpoint directory: {checkpoint_dir}")
            accelerator.print(f"📊 Training State from Metrics:")
            if last_epoch is not None:
                accelerator.print(f"   - Last Completed Epoch: {last_epoch}")
                accelerator.print(f"   - Resume from Epoch: {start_epoch} (next epoch after saved)")
            else:
                accelerator.print(f"   - No epoch found in metrics, starting from Epoch 0")
                start_epoch = 0
            accelerator.print(f"   - Best Loss (from metrics): {best_loss:.4f}")
            accelerator.print(f"   ✅ Final Best Loss: {best_loss:.4f} (will be used for best model comparison)")
            
            if last_metrics:
                accelerator.print(f"📈 Last Training Performance:")
                if last_metrics.get("train"):
                    train_metrics = last_metrics["train"]
                    accelerator.print(f"   - Train Loss: {train_metrics.get('train_loss', 'N/A'):.4f}")
                    accelerator.print(f"   - Train Acc: {train_metrics.get('train_acc', 'N/A'):.4f}")
                    accelerator.print(f"   - Train PTM Acc: {train_metrics.get('train_ptm_acc', 'N/A'):.4f}")
                    accelerator.print(f"   - Train PPL: {train_metrics.get('train_preplexity', 'N/A'):.2f}")
                
                if last_metrics.get("val"):
                    val_metrics = last_metrics["val"]
                    accelerator.print(f"   - Val Loss: {val_metrics.get('val_loss', 'N/A'):.4f}")
                    accelerator.print(f"   - Val Acc: {val_metrics.get('val_acc', 'N/A'):.4f}")
                    accelerator.print(f"   - Val PTM Acc: {val_metrics.get('val_ptm_acc', 'N/A'):.4f}")
                    accelerator.print(f"   - Val PPL: {val_metrics.get('val_preplexity', 'N/A'):.2f}")
                
                if last_metrics.get("test"):
                    test_metrics = last_metrics["test"]
                    accelerator.print(f"   - Test Loss: {test_metrics.get('test_loss', 'N/A'):.4f}")
                    accelerator.print(f"   - Test Acc: {test_metrics.get('test_acc', 'N/A'):.4f}")
                    accelerator.print(f"   - Test PTM Acc: {test_metrics.get('test_ptm_acc', 'N/A'):.4f}")
            
            accelerator.print(f"🎯 Training will continue from Epoch {start_epoch} (displayed as Epoch {start_epoch + 1}/{train_args.get('num_train_epochs', 10)}) to Epoch {train_args.get('num_train_epochs', 10)}")
            accelerator.print("=" * 80)
    else:
        start_epoch = 0
        start_step = 0
        best_loss = float("inf")
        if accelerator.is_local_main_process:
            accelerator.print("🚀 Starting new training from scratch")

    # Embeddings are already loaded in the dataset, no need for separate loader
    
    model_to_save = model if accelerator.distributed_type==DistributedType.NO else model.module
    total_steps = start_step
    
    # Debug: Print epoch range
    num_epochs = train_args.get("num_train_epochs", 10)
    if accelerator.is_local_main_process:
        accelerator.print(f"🔍 Debug Info:")
        accelerator.print(f"   - start_epoch: {start_epoch}")
        accelerator.print(f"   - num_epochs: {num_epochs}")
        accelerator.print(f"   - Epoch range: range({start_epoch}, {num_epochs}) = {list(range(start_epoch, num_epochs))[:10]}...")
        accelerator.print(f"   - Total epochs to train: {num_epochs - start_epoch}")
    
    for epoch in range(start_epoch, num_epochs):
        # Create progress bar for training (only on main process)
        train_pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{train_args.get('num_train_epochs', 10)}",
            disable=not accelerator.is_local_main_process,
            leave=True,
        )
        
        for batch in train_pbar:
            optimizer.zero_grad()
            
            # Get data from batch (already tokenized in collate_fn)
            # Embeddings 已经是 float16，只需传输到 device，保持原有精度
            embeddings = batch["embeddings"].to(device=device, non_blocking=True)  # (batch_size, max_seq_len, embed_dim) float16
            pad_mask = batch["pad_mask"].to(device, non_blocking=True)  # (batch_size, max_seq_len)
            original_input_ids = batch["original_input_ids"].to(device, non_blocking=True)  # (batch_size, max_seq_len)
            ptm_input_ids = batch["ptm_input_ids"].to(device, non_blocking=True)  # (batch_size, max_seq_len)
            if "position_mask" in batch:
                position_mask = batch["position_mask"].to(device, non_blocking=True)  # (batch_size, max_seq_len)
            else:
                position_mask = None

            # Get functional role data if available
            functional_role = batch.get("functional_role", None)
            functional_role_position = batch.get("functional_role_position", None)
            if functional_role is not None:
                functional_role = functional_role.to(device, non_blocking=True)  # (batch_size, max_seq_len)
                functional_role_position = functional_role_position.to(device, non_blocking=True)  # (batch_size, max_seq_len)
            
            # 🚀 纯 FP16: 直接前向传播 (模型参数已经是 float16)
            # Forward pass through model
            outputs = model(embeddings=embeddings, position_mask=position_mask)  # Dict[str, torch.Tensor] - {head_name: logits}

            # Get the actual model (handle accelerator wrapping)
            actual_model = model.module if hasattr(model, 'module') else model

            # Compute loss for each head using model's compute_loss method
            losses_compute_related = {}
            for head_type, logits in outputs.items():
                loss_compute_related = {
                    "logits": logits,
                    "kwargs": {
                        "device": device,
                    }
                }
                if head_type == "original":
                    loss_compute_related["target"] = original_input_ids
                elif head_type == "ptm":
                    loss_compute_related["target"] = ptm_input_ids
                elif head_type == "functional_role":
                    loss_compute_related["target"] = functional_role

                losses_compute_related[head_type] = loss_compute_related

            # Compute losses using model's compute_loss method
            losses = actual_model.compute_loss(losses_compute_related)
            total_loss = losses["total"]

            # Compute accuracy for each head
            head_accs = {}
            for head_type, logits in outputs.items():
                if head_type == "original":
                    # Accuracy at all non-padding positions
                    non_padding_mask = pad_mask & (original_input_ids != 0)
                    if non_padding_mask.any():
                        acc = (logits.argmax(dim=-1) == original_input_ids)[non_padding_mask].float().mean()
                    else:
                        acc = torch.tensor(0.0, device=device)
                elif head_type == "ptm":
                    # Accuracy at PTM positions only
                    acc = (logits.argmax(dim=-1) == ptm_input_ids).float().mean()
                else:
                    acc = torch.tensor(0.0, device=device)
                head_accs[head_type] = acc
            
            # Get accuracies for logging
            acc = head_accs.get("original", torch.tensor(0.0, device=device))
            ptm_acc = head_accs.get("ptm", torch.tensor(0.0, device=device))
            
            # Save batch size and sequence length for logging before cleanup
            batch_size = embeddings.shape[0]
            seq_len = embeddings.shape[1]
            
            # 🚀 AMP: 使用 Accelerate 的混合精度训练
            accelerator.backward(total_loss)

            # ✅ 梯度裁剪：在 step 前防止梯度爆炸
            accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # 🚀 AMP: Accelerate 会自动处理 scaler
            optimizer.step()

            # 📊 修复 perplexity 计算：FP32 + clamp 防止溢出
            preplexity = torch.exp(torch.clamp(total_loss.detach().float(), max=10))

            # 🔊 调试：监控梯度范数 (前几步)
            if total_steps <= 2 and accelerator.is_local_main_process:
                # 计算梯度范数用于调试
                total_norm = 0
                param_count = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                        param_count += 1
                total_norm = total_norm ** (1. / 2)
                print(f"AMP Debug: grad_norm = {total_norm:.6f} (from {param_count} params)")

            # update learning rate
            scheduler.step()
            
            # Update progress bar with metrics (only on main process)
            # Get current learning rate from optimizer (after scheduler.step())
            current_lr = optimizer.param_groups[0]['lr'] if accelerator.is_local_main_process else 0.0
            if accelerator.is_local_main_process:
                train_pbar.set_postfix({
                    "loss": f"{total_loss.item():.4f}",
                    "acc": f"{acc.item():.4f}",
                    "ptm_acc": f"{ptm_acc.item():.4f}",
                    "ppl": f"{preplexity.item():.2f}",
                    "lr": f"{current_lr:.2e}",
                })
                logger.log(
                    {
                        "Epoch": epoch,
                        "Step": total_steps,
                        "train_loss": total_loss.item(),
                        "train_preplexity": preplexity.item(),
                        "train_acc": acc.item(),
                        "train_ptm_acc": ptm_acc.item(),
                        "act_bs": batch_size,
                        "act_seq_len": seq_len,
                    }
                )
            total_steps += 1
            
            # Clean up batch variables after logging
            del outputs, losses_compute_related, losses, head_accs, preplexity, acc, ptm_acc, total_loss
            del embeddings, pad_mask, original_input_ids, ptm_input_ids
            
            # Removed frequent empty_cache() calls - only clear at epoch/validation end
            
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
                        # Get data from batch (already tokenized in collate_fn)
                        # Embeddings 已经是 float16，只需传输到 device，保持原有精度
                        embeddings = val_batch["embeddings"].to(device=device, non_blocking=True)  # (batch_size, max_seq_len, embed_dim) float16
                        pad_mask = val_batch["pad_mask"].to(device, non_blocking=True)  # (batch_size, max_seq_len)
                        original_input_ids = val_batch["original_input_ids"].to(device, non_blocking=True)  # (batch_size, max_seq_len)
                        ptm_input_ids = val_batch["ptm_input_ids"].to(device, non_blocking=True)  # (batch_size, max_seq_len)

                        # Get functional role data if available
                        functional_role = val_batch.get("functional_role", None)
                        functional_role_position = val_batch.get("functional_role_position", None)
                        if functional_role is not None:
                            functional_role = functional_role.to(device, non_blocking=True)  # (batch_size, max_seq_len)
                            functional_role_position = functional_role_position.to(device, non_blocking=True)  # (batch_size, max_seq_len)

                        
                        # 🚀 纯 FP16: 验证前向传播
                        # Forward pass through model
                        outputs = model(embeddings=embeddings)  # Dict[str, torch.Tensor]

                        # Get the actual model (handle accelerator wrapping)
                        actual_model = model.module if hasattr(model, 'module') else model

                        # Compute loss for each head using model's compute_loss method
                        losses_compute_related = {}
                        for head_type, logits in outputs.items():
                            loss_compute_related = {
                                "logits": logits,
                                "kwargs": {
                                    "device": device,
                                }
                            }
                            if head_type == "original":
                                loss_compute_related["target"] = original_input_ids
                            elif head_type == "ptm":
                                loss_compute_related["target"] = ptm_input_ids
                            elif head_type == "functional_role":
                                loss_compute_related["target"] = functional_role
                            losses_compute_related[head_type] = loss_compute_related

                        # Compute losses using model's compute_loss method
                        losses = actual_model.compute_loss(losses_compute_related)
                        loss = losses["total"]

                        # Compute accuracy for each head
                        original_logits = outputs.get("original", None)
                        ptm_logits = outputs.get("ptm", None)
                        
                        # Accuracy on original (at non-padding positions)
                        non_padding_mask = pad_mask & (original_input_ids != 0)
                        if non_padding_mask.any() and original_logits is not None:
                            acc = (original_logits.argmax(dim=-1) == original_input_ids)[non_padding_mask].float().mean()
                        else:
                            acc = torch.tensor(0.0, device=device)
                        
                        # Compute PTM accuracy on ptm at PTM sites
                        if ptm_logits is not None:
                            ptm_acc = (ptm_logits.argmax(dim=-1) == ptm_input_ids).float().mean()
                        else:
                            ptm_acc = torch.tensor(0.0, device=device)
                    
                    # compute perplexity
                    preplexity = torch.exp(loss)
                    
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
                    
                    # Clean up validation batch variables
                    del embeddings, pad_mask, original_input_ids, ptm_input_ids
                    del outputs, losses_compute_related, losses, original_logits, ptm_logits
                    del acc, ptm_acc, preplexity, loss, non_padding_mask
                
                # Clean up validation variables and clear cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
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
                                {
                                    "model": model_to_save.state_dict(),
                                    "config": model_config,
                                    "optimizer": optimizer.state_dict() if optimizer is not None else None,
                                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                                    "training_state": {
                                        "epoch": epoch,
                                        "step": total_steps,
                                        "best_loss": best_loss,
                                    }
                                },
                                best_ckpt_path,
                            )
                            accelerator.print(f"💾 Best model saved! (val_loss: {avg_val_loss:.4f})")
                if accelerator.is_local_main_process:
                    # save last model with training state
                    torch.save(
                        {
                            "model": model_to_save.state_dict(),
                            "config": model_config,
                            "optimizer": optimizer.state_dict() if optimizer is not None else None,
                            "scheduler": scheduler.state_dict() if scheduler is not None else None,
                            "training_state": {
                                "epoch": epoch,
                                "step": total_steps,
                                "best_loss": best_loss,
                            }
                        },
                        last_ckpt_path,
                    )
                    accelerator.print(f"Epoch {epoch}, Step {total_steps} finished!")
                
                # Clean up validation variables
                del val_losses, val_ppls, val_accs, val_ptm_accs
                model.train()  # Set model back to training mode
                
                # Clear CUDA cache after validation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Clean up epoch-level variables and clear cache at end of each epoch
        del train_pbar
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Force garbage collection to help with memory fragmentation
            gc.collect()
    
    if accelerator.is_local_main_process:
        # save last model with training state
        torch.save(
            {
                "model": model_to_save.state_dict(),
                "config": model_config,
                "optimizer": optimizer.state_dict() if optimizer is not None else None,
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "training_state": {
                    "epoch": num_epochs - 1,
                    "step": total_steps,
                    "best_loss": best_loss,
                }
            },
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
                    # Get data from batch (already tokenized in collate_fn)
                    # Embeddings 已经是 float16，只需传输到 device，保持原有精度
                    embeddings = test_batch["embeddings"].to(device=device, non_blocking=True)  # (batch_size, max_seq_len, embed_dim) float16
                    pad_mask = test_batch["pad_mask"].to(device, non_blocking=True)  # (batch_size, max_seq_len)
                    original_input_ids = test_batch["original_input_ids"].to(device, non_blocking=True)  # (batch_size, max_seq_len)
                    ptm_input_ids = test_batch["ptm_input_ids"].to(device, non_blocking=True)  # (batch_size, max_seq_len)

                    # Get functional role data if available
                    functional_role = test_batch.get("functional_role", None)
                    functional_role_position = test_batch.get("functional_role_position", None)
                    if functional_role is not None:
                        functional_role = functional_role.to(device, non_blocking=True)  # (batch_size, max_seq_len)
                        functional_role_position = functional_role_position.to(device, non_blocking=True)  # (batch_size, max_seq_len)
                    
                    # 🚀 纯 FP16: 测试前向传播
                    # Forward pass through model
                    outputs = model(embeddings=embeddings)  # Dict[str, torch.Tensor]

                    # Get the actual model (handle accelerator wrapping)
                    actual_model = model.module if hasattr(model, 'module') else model

                    # Compute loss for each head using model's compute_loss method
                    losses_compute_related = {}
                    for head_type, logits in outputs.items():
                        loss_compute_related = {
                            "logits": logits,
                            "kwargs": {
                                "device": device,
                            }
                        }
                        if head_type == "original":
                            loss_compute_related["target"] = original_input_ids
                        elif head_type == "ptm":
                            loss_compute_related["target"] = ptm_input_ids
                        elif head_type == "functional_role":
                            loss_compute_related["target"] = functional_role

                        losses_compute_related[head_type] = loss_compute_related

                    # Compute losses using model's compute_loss method
                    losses = actual_model.compute_loss(losses_compute_related)
                    loss = losses["total"]

                    
                    # Compute accuracy for each head
                    original_logits = outputs.get("original", None)
                    ptm_logits = outputs.get("ptm", None)
                    
                    # Accuracy on original (at non-padding positions)
                    non_padding_mask = pad_mask & (original_input_ids != 0)
                    if non_padding_mask.any() and original_logits is not None:
                        acc = (original_logits.argmax(dim=-1) == original_input_ids)[non_padding_mask].float().mean()
                    else:
                        acc = torch.tensor(0.0, device=device)
                    
                    # Compute PTM accuracy on ptm at PTM sites
                    acc = (logits.argmax(dim=-1) == ptm_input_ids).float().mean()
                
                # compute perplexity
                preplexity = torch.exp(loss)
                
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
                            "Epoch": num_epochs - 1,
                            "Step": total_steps,
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
                        "Epoch": num_epochs - 1,
                        "Step": total_steps,
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
    parser = argparse.ArgumentParser(description="PTM-Mamba Training (Memmap Format)")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to configuration YAML file",
    )
    
    # Use parse_known_args to capture unknown arguments (like Hydra-style key=value)
    args, unknown_args = parser.parse_known_args()
    
    # Load configuration
    cfg = load_config(args.config)
    
    # Parse unknown_args for Hydra-style key=value format
    for arg in unknown_args:
        if "=" not in arg:
            # Unknown argument without =, warn user
            print(f"⚠️  Warning: Unknown argument '{arg}'. If you meant to override a config, use 'key=value' format.")
            continue
        
        # This is a Hydra-style override: key=value
        key_path, value = arg.split("=", 1)
        keys = key_path.split(".")
        
        # Navigate to the nested dict and set the value
        current = cfg
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Convert value to appropriate type
        final_key = keys[-1]
        try:
            # Try to convert to number
            if "." in value and "e" not in value.lower():
                # Float (but not scientific notation)
                current[final_key] = float(value)
            elif "e" in value.lower():
                # Scientific notation (e.g., 4e-4)
                current[final_key] = float(value)
            else:
                # Try int first
                current[final_key] = int(value)
        except ValueError:
            # Keep as string, but handle boolean strings
            if value.lower() == "true":
                current[final_key] = True
            elif value.lower() == "false":
                current[final_key] = False
            elif value.lower() == "null" or value.lower() == "none":
                current[final_key] = None
            else:
                current[final_key] = value
        
        print(f"✅ Override: {key_path} = {current[final_key]}")
    
    # Access configuration groups
    data_config = cfg["dataset"]
    train_args = cfg["training"]
    model_config = cfg["model"]
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    # Now Accelerator initializes and can only see GPUs specified in CUDA_VISIBLE_DEVICES
    accelerator = Accelerator(
        kwargs_handlers=[kwargs],
        mixed_precision="bf16"  # 启用内置 AMP + GradScaler
    )
    device = accelerator.device
    set_seed(cfg.get("seed", 42))
    
    # Print GPU usage summary
    if torch.cuda.is_available() and accelerator.is_local_main_process:
        num_processes = accelerator.num_processes
        print(f"GPUs in use: {list(range(num_processes))}")
    
    # Check if resuming from output directory
    resume_from_output = train_args.get("resume_from_output", None)
    
    # Create output directory with timestamp (only on main process)
    # All outputs (config, checkpoints, metrics) will be saved here
    if resume_from_output and os.path.exists(resume_from_output):
        # Use existing output directory when resuming
        output_dir = resume_from_output
        if accelerator.is_local_main_process:
            print(f"📂 Resuming in existing output directory: {output_dir}")
    else:
        # Create new output directory
        exp_name = cfg.get("exp_name", "ptm-mamba-memmap")
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        output_dir = f"outputs/{exp_name}-{timestamp}"
        
        # All processes need to know the output_dir path
        # Main process creates directory and saves config, other processes just ensure it exists
        os.makedirs(output_dir, exist_ok=True)
        
        if accelerator.is_local_main_process:
            # Save config to output directory (only main process)
            config_output_path = os.path.join(output_dir, "config.yaml")
            with open(config_output_path, 'w', encoding='utf-8') as f:
                yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
            print(f"📝 Config saved to: {config_output_path}")
            print(f"💾 Output directory: {output_dir}")
    
    if "wandb" in cfg.get("report_to", []):
        import wandb

        if accelerator.is_local_main_process:
            wandb.init(
                project="PTM-Mamba", config=dict(cfg), name=output_dir
            )
        logger = wandb
    else:
        # Initialize logger with output_dir and config
        # If resuming from output directory, set resume=True to load existing metrics
        logger_save_dir = output_dir if accelerator.is_local_main_process else None
        is_resuming = resume_from_output is not None and os.path.exists(resume_from_output) if resume_from_output else False
        logger = TrainLogger(
            save_dir=logger_save_dir,
            config=cfg if accelerator.is_local_main_process else None,
            resume=is_resuming
        )

    tokenizer = PTMTokenizer()
    seed = cfg.get("seed", None)
    
    # Get dataset directory (memmap directory)
    dataset_dir = data_config.get("dataset_dir", None)
    if dataset_dir is None:
        raise ValueError("dataset_dir must be provided in dataset config. This should point to the directory containing memmap files (embeddings_combined_memmap)")
    
    # Get split parameters
    val_size = data_config.get("val_size", 0)
    test_size = data_config.get("test_size", 0)
    
    # Get preload_all flag (from command line or config)
    preload_all = data_config.get("preload_all", False)

    # Get use_functional_role flag (from command line or config)
    use_functional_role = data_config.get("use_functional_role", False)
    
    # Load dataset from memmap format and split
    if accelerator.is_local_main_process:
        mode_str = "preload mode" if preload_all else "memmap mode"
        print(f"🚀 Loading dataset from memmap format ({mode_str}): {dataset_dir}")
    dataset = get_ptm_dataset_memmap(
        dataset_dir=dataset_dir,
        device=torch.device('cpu'),  # 数据先放在CPU，让pin_memory处理GPU传输
        seed=seed,
        val_size=val_size,
        test_size=test_size,
        preload_all=preload_all,
        use_functional_role=use_functional_role,
    )
    
    # Log dataset split information and save split mapping (only on main process)
    if accelerator.is_local_main_process:
        dataset_info = {
            "train_size": len(dataset["train"]) if "train" in dataset and dataset["train"] is not None else 0,
            "val_size": len(dataset["val"]) if "val" in dataset and dataset["val"] is not None else 0,
            "test_size": len(dataset["test"]) if "test" in dataset and dataset["test"] is not None else 0,
        }
        logger.log({"dataset_split": dataset_info})
        accelerator.print(f"📊 Dataset split - Train: {dataset_info['train_size']}, Val: {dataset_info['val_size']}, Test: {dataset_info['test_size']}")
        
        split_mapping = dataset.get("split_mapping", {})
        
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

    # Auto-detect esm_embed_dim from ESM model name (must be done before model initialization)
    # ESM model is always used, so we always need to set esm_embed_dim
    esm_model_name = train_args.get("esm_model", "esm2_15b")
    repr_layer_override = train_args.get("repr_layer", None)
    try:
        embed_dim = get_esm_embed_dim(esm_model_name, repr_layer_override)
        if accelerator.is_local_main_process:
            accelerator.print(f"✅ Auto-detected ESM embedding dimension: {embed_dim} (from {esm_model_name})")
    except Exception as e:
        if accelerator.is_local_main_process:
            accelerator.print(f"❌ Failed to auto-detect ESM embedding dimension: {e}")
            accelerator.print(f"⚠️  Falling back to default: 5120 (esm2_15b)")
        embed_dim = 5120  # Default fallback
    
    # Get model configuration
    vocab_size = tokenizer.get_vocab_size()
    d_model = model_config.get("d_model", 512)
    block_config = model_config.get("block_config", {"type": "lora"})
    # Heads config: 'type' is used both for registry lookup and as key in model.heads
    heads_config = model_config.get("heads_config", [
        {"type": "original"},
        {"type": "ptm"},
    ])
    
    # Handle checkpoint loading from output directory
    if resume_from_output:
        # Resume from output directory (always loads from last.ckpt)
        # Note: epoch and best_loss will be loaded from metrics.json in train() function
        ckpt_path = os.path.join(resume_from_output, "last.ckpt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model_config_dict = ckpt["config"]
        
        # Initialize PTMModel with saved config
        model = PTMModel(
            embed_dim=embed_dim,
            vocab_size=vocab_size,
            d_model=model_config_dict.get("d_model", d_model),
            block_config=model_config_dict.get("block_config", block_config),
            heads_config=model_config_dict.get("heads_config", heads_config),
            device=device,
            dtype=torch.float16,  # 统一使用 float16 精度
        )
        
        # Load state dict
        model.load_state_dict(ckpt["model"], strict=True)
        # 检查 checkpoint 参数类型，仅在需要时转换（避免不必要的转换）
        first_param = next(model.parameters())
        if first_param.dtype != torch.float16:
            # Checkpoint 是 float32，需要转换为 float16（仅转换一次）
            model = model.to(dtype=torch.float16)
            accelerator.print(f"✅ Model loaded from output directory: {resume_from_output}")
            accelerator.print(f"   Checkpoint was float32, converted to float16")
        else:
            # Checkpoint 已经是 float16，无需转换
            accelerator.print(f"✅ Model loaded from output directory: {resume_from_output}")
            accelerator.print(f"   Checkpoint already in float16, no conversion needed")
    else:
        # Initialize PTMModel with block and heads configuration
        model = PTMModel(
            embed_dim=embed_dim,
            vocab_size=vocab_size,
            d_model=d_model,
            block_config=block_config,
            heads_config=heads_config,
            device=device,
            dtype=torch.float32,  # FP32 权重，让 AMP 处理前向精度
        )
    # 模型初始化时已指定 dtype=torch.float16，参数自动为 float16，无需转换
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    accelerator.print(f"Number of parameters: {num_params:,}")
    accelerator.print(f"Model precision: {next(model.parameters()).dtype}")
    
    # Create data loaders - memmap格式直接使用orig_ids和ptm_ids，不需要tokenize
    def collate_fn(batch):
        """
        优化后的 collate function：
        1. Stacks embeddings (保持 float16，避免不必要的转换)
        2. 批量创建 pad_mask 和输入 tensors
        3. 移除字符串对象以提高性能
        """
        max_seq_len = batch[0]["embeddings"].shape[0]  # 所有样本都有相同形状

        # 批量堆叠 embeddings（保持 float16 CPU tensor）
        embeddings = torch.stack([item["embeddings"] for item in batch])  # (batch_size, max_seq_len, embed_dim)

        # 收集序列长度
        seq_lengths = [item["seq_length"] for item in batch]

        # 批量创建 pad_mask（向量化操作，比循环快）
        seq_lengths_tensor = torch.tensor(seq_lengths, dtype=torch.long)
        pad_mask = torch.arange(max_seq_len, device='cpu').unsqueeze(0) < seq_lengths_tensor.unsqueeze(1)  # (batch_size, max_seq_len)

        # 批量创建输入 tensors（避免逐个 copy）
        orig_ids_list = [item["orig_ids"] for item in batch]
        ptm_ids_list = [item["ptm_ids"] for item in batch]

        # 使用 torch.stack 而不是逐个 from_numpy（更高效）
        original_input_ids = torch.stack([torch.from_numpy(ids) for ids in orig_ids_list]).long()
        ptm_input_ids = []
        for ids in ptm_ids_list:
            # 复制数组以确保可写（memmap 数组可能是只读的）
            ids_copy = ids.copy()
            mask = ~tokenizer.is_ptm_token(ids_copy)
            ids_copy[mask] = 59
            ptm_input_ids.append(torch.from_numpy(ids_copy))
        ptm_input_ids = torch.stack(ptm_input_ids).long()

        # 检查是否有functional role数据
        has_functional_role = "functional_role" in batch[0]
        if has_functional_role:
            functional_role_list = [item["functional_role"] for item in batch]
            functional_role_position_list = [item["functional_role_position"] for item in batch]

            # 使用 torch.stack 处理functional role数据
            functional_role = torch.stack([torch.from_numpy(fr) for fr in functional_role_list]).float()
            functional_role_position = torch.stack([torch.from_numpy(frp) for frp in functional_role_position_list]).long()

            # 创建 position_mask: functional_role_position > 0 表示有functional role的位置
            position_mask = (functional_role_position > 0)

        # 基础返回结构（移除字符串对象）
        result = {
            "embeddings": embeddings,  # float16 CPU tensor
            "pad_mask": pad_mask,      # bool CPU tensor
            "original_input_ids": original_input_ids,  # long CPU tensor
            "ptm_input_ids": ptm_input_ids,           # long CPU tensor
            "seq_lengths": seq_lengths,               # list of int
        }

        # 可选：只在需要时添加 unique_ids（从 protein_idx 映射）
        protein_indices = [item.get("protein_idx") for item in batch]
        if all(pid is not None for pid in protein_indices):
            result["protein_indices"] = protein_indices

        # 如果有functional role数据，添加到结果中
        if has_functional_role:
            result["functional_role"] = functional_role  # float CPU tensor
            result["functional_role_position"] = functional_role_position  # long CPU tensor
            result["position_mask"] = position_mask  # bool CPU tensor

        return result
    
    # 🚀 优化 DataLoader 配置：多 worker + prefetch + persistent workers
    num_workers = 4 if not preload_all else 0  # 预加载模式用单线程，避免重复加载
    train_loader = DataLoader(
        dataset["train"],
        batch_size=train_args["per_device_train_batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,  # 增加预取因子
    )
    val_loader = DataLoader(
        dataset["val"],
        batch_size=train_args["per_device_train_batch_size"],
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    
    # Create test loader if test dataset exists
    test_loader = None
    if "test" in dataset and dataset["test"] is not None and len(dataset["test"]) > 0:
        test_loader = DataLoader(
            dataset["test"],
            batch_size=train_args["per_device_train_batch_size"],
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True if device.type == 'cuda' else False,
            persistent_workers=num_workers > 0,
            prefetch_factor=4 if num_workers > 0 else None,
        )
    # Ensure lr is a float (YAML may read scientific notation as string)
    lr = float(train_args["lr"]) if isinstance(train_args["lr"], str) else train_args["lr"]
    
    optimizer = torch.optim.AdamW(
        model.parameters(), lr, betas=(0.9, 0.98), weight_decay=0.01
    )

    scheduler = Esm2LRScheduler(
        optimizer, last_epoch=-1, init_lr=lr, on_use=False
    )

    # 🚀 纯 float16 训练：不使用 GradScaler（因为模型参数已经是 float16）
    scaler = None
    if accelerator.is_local_main_process:
        accelerator.print(f"🔥 AMP enabled: {accelerator.mixed_precision} precision training")

    # Prepare test_loader if it exists
    if test_loader is not None:
        model, optimizer, train_loader, val_loader, test_loader = accelerator.prepare(
            model, optimizer, train_loader, val_loader, test_loader
        )
    else:
        model, optimizer, train_loader, val_loader = accelerator.prepare(
            model, optimizer, train_loader, val_loader
        )
    # Note: scheduler and scaler are handled manually since accelerator may not support them
    
    # If resuming from output directory, restore optimizer and scheduler state after preparing
    if resume_from_output:
        # Load checkpoint again to restore optimizer and scheduler (after they're prepared)
        ckpt_path = os.path.join(resume_from_output, "last.ckpt")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location='cpu')
            if "optimizer" in ckpt and ckpt["optimizer"] is not None:
                optimizer.load_state_dict(ckpt["optimizer"])
                if accelerator.is_local_main_process:
                    accelerator.print(f"✅ Optimizer state restored")
            if "scheduler" in ckpt and ckpt["scheduler"] is not None:
                scheduler.load_state_dict(ckpt["scheduler"])
                if accelerator.is_local_main_process:
                    accelerator.print(f"✅ Scheduler state restored")

    train(
        config_dict=cfg,
        model_config=model_config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        tokenizer=tokenizer,
        logger=logger,
        accelerator=accelerator,
        checkpoint_dir=output_dir,
        test_loader=test_loader,
    )

      
if __name__ == "__main__":
    main()


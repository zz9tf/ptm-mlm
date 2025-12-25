import torch
from types import SimpleNamespace
from models.mamba.lm import MambaLMHeadModel
from accelerate import Accelerator
import os


def load_ckpt(ckpt_path, tokenizer, accelerator_or_device, optimizer=None, scheduler=None):
    """
    Load checkpoint and create model, optionally restore optimizer and scheduler state.
    @param ckpt_path: Path to checkpoint file.
    @param tokenizer: Tokenizer instance.
    @param accelerator_or_device: Accelerator instance or torch.device/str for device management.
    @param optimizer: Optimizer instance (optional, for restoring optimizer state).
    @param scheduler: Scheduler instance (optional, for restoring scheduler state).
    @returns: Tuple of (model, training_state_dict) where training_state_dict contains epoch, step, best_loss, etc.
    """
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model_state_dict = ckpt["model"]
    # Convert dict config to object (like in train.py)
    model_config_dict = ckpt["config"]
    model_config = SimpleNamespace(**model_config_dict)
    model_config.vocab_size = tokenizer.get_vocab_size()
    
    # Support both Accelerator and device (str/torch.device)
    if isinstance(accelerator_or_device, Accelerator):
        device = accelerator_or_device.device
    elif isinstance(accelerator_or_device, (str, torch.device)):
        device = torch.device(accelerator_or_device) if isinstance(accelerator_or_device, str) else accelerator_or_device
    else:
        raise TypeError(f"accelerator_or_device must be Accelerator, str, or torch.device, got {type(accelerator_or_device)}")
    
    model = MambaLMHeadModel(config=model_config, device=device)
    msg = model.load_state_dict(model_state_dict, strict=True)
    print(msg)
    
    # Extract training state if available
    training_state = ckpt.get("training_state", {})
    
    # Restore optimizer state if provided and available
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
        print("✅ Optimizer state restored")
    
    # Restore scheduler state if provided and available
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
        print("✅ Scheduler state restored")
    
    return model, training_state


def load_ckpt_from_output_dir(output_dir, tokenizer, accelerator_or_device, optimizer=None, scheduler=None):
    """
    Load checkpoint from output directory (always loads from last.ckpt).
    @param output_dir: Path to output directory containing checkpoints.
    @param tokenizer: Tokenizer instance.
    @param accelerator_or_device: Accelerator instance or torch.device/str for device management.
    @param optimizer: Optimizer instance (optional, for restoring optimizer state).
    @param scheduler: Scheduler instance (optional, for restoring scheduler state).
    @returns: Tuple of (model, training_state_dict, checkpoint_path).
    """
    ckpt_path = os.path.join(output_dir, "last.ckpt")
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    print(f"📂 Loading checkpoint from output directory: {ckpt_path}")
    model, training_state = load_ckpt(ckpt_path, tokenizer, accelerator_or_device, optimizer, scheduler)
    
    return model, training_state, ckpt_path


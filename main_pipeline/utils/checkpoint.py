import torch
from types import SimpleNamespace
from models.mamba.lm import MambaLMHeadModel
from accelerate import Accelerator


def load_ckpt(ckpt_path, tokenizer, accelerator_or_device):
    """
    Load checkpoint and create model.
    @param ckpt_path: Path to checkpoint file.
    @param tokenizer: Tokenizer instance.
    @param accelerator_or_device: Accelerator instance or torch.device/str for device management.
    @returns: Loaded model.
    """
    ckpt = torch.load(ckpt_path)
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
    return model


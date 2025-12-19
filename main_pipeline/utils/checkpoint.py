import torch
from models.mamba.lm import MambaLMHeadModel
from accelerate import Accelerator


def load_ckpt(ckpt_path, tokenizer, accelerator):
    """
    Load checkpoint and create model.
    @param ckpt_path: Path to checkpoint file.
    @param tokenizer: Tokenizer instance.
    @param accelerator: Accelerator instance for device management.
    @returns: Loaded model.
    """
    ckpt = torch.load(ckpt_path)
    model_state_dict = ckpt["model"]
    model_config = ckpt["config"]
    model_config.vocab_size = tokenizer.get_vocab_size()
    model = MambaLMHeadModel(config=model_config, device=accelerator.device)
    msg = model.load_state_dict(model_state_dict, strict=True)
    print(msg)
    return model


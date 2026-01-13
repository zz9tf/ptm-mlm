"""
Adaptor inference script for processing embeddings.
This script loads adapter models (LoRA, etc.) and processes input embeddings.

This is a shared module used by all downstream tasks.
"""
import torch
import json
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

from main_pipeline.models.model import PTMModel

class AdaptorInference:
    """
    Adaptor inference class for processing embeddings.
    If checkpoint_path is provided, loads adapter model (LoRA, etc.) and processes embeddings through block.
    If checkpoint_path is None, directly uses pretrained model embeddings (no special token removal or merging).
    """
    
    def __init__(self, checkpoint_path: str = None, device: str = None, embed_dim: int = 1152):
        """
        Initialize adaptor inference model.

        @param checkpoint_path: Path to trained adapter model checkpoint (.ckpt file).
                                If None, no model is loaded and pretrained embeddings are used directly.
        @param device: Device to run inference on (None for auto-detection)
        @param embed_dim: Input embedding dimension (default 1152, for ESM-C 600M)
        """
        # 确定设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.use_block = checkpoint_path is not None
        
        if self.use_block:
            # Load adaptor checkpoint
            print(f"📦 Loading adapter model from {checkpoint_path}...")
            ckpt = torch.load(checkpoint_path, map_location='cpu')
            model_state_dict = ckpt["model"]
            model_config_dict = ckpt["config"]

            # Get model parameters from config
            embed_dim = model_config_dict.get("embed_dim", embed_dim)  # Use embed_dim from config as default
            vocab_size = model_config_dict.get("vocab_size", 32)  # Adapter doesn't need vocab
            d_model = model_config_dict.get("d_model", 512)
            block_config = model_config_dict.get("block_config", {"type": "lora"})

            # Initialize PTMModel (only use block, heads are not created)
            self.model = PTMModel(
                embed_dim=embed_dim,
                vocab_size=vocab_size,
                d_model=d_model,
                block_config=block_config,
                heads_config=[],  # Don't create heads, we only need block output
                device=self.device,
            )

            # Load model state
            msg = self.model.load_state_dict(model_state_dict, strict=False)
            print(f"📝 Model loading info: {msg}")

            # Ensure model is on the correct device
            self.model = self.model.to(self.device)
            self.model.eval()

            # Get hidden size from model config
            self.hidden_size = d_model
            self.embed_dim = embed_dim

            print(f"✅ Adapter model loaded successfully! Input dim: {embed_dim}, Output dim: {d_model}")
            print(f"🔧 Mode: Adaptor Block (accepts embeddings input, generates adapted output)")
        else:
            # Don't use block, directly use pretrained model embeddings
            self.model = None
            self.embed_dim = embed_dim
            # hidden_size will be inferred from embeddings during processing
            print(f"✅ Using pretrained model embeddings directly (no block processing)")
            print(f"🔧 Mode: Direct Embeddings (no special token removal or merging)")
    
    
    @torch.no_grad()
    def process_embeddings(self, embeddings_tensor: torch.Tensor, metadata_list: list):
        """
        Batch process embeddings, focused on batch processing.
        
        Responsibilities:
        - If checkpoint_path is provided: Process entire batch tensor through block
        - If checkpoint_path is None: Return embeddings directly (move to CPU)
        
        ⚠️ Note:
        - This method only handles batch processing, does not merge or remove special tokens
        - Merging and special token removal are handled by Pipeline
        - Input is already padded batch tensor

        @param embeddings_tensor: Tensor (batch_size, max_seq_len, embed_dim)
        @param metadata_list: List of metadata dicts (required)
        @returns: (processed_embeddings_tensor, processed_metadata_list)
                 - processed_embeddings_tensor: Tensor (batch_size, max_seq_len, hidden_size)
                 - processed_metadata_list: List of metadata dicts (same as input)
        """
        
        if not isinstance(embeddings_tensor, torch.Tensor) or embeddings_tensor.dim() != 3:
            raise ValueError(
                f"Expected 3D tensor (batch_size, max_seq_len, embed_dim), "
                f"got {type(embeddings_tensor)} with dim {embeddings_tensor.dim() if isinstance(embeddings_tensor, torch.Tensor) else 'N/A'}"
            )
        # Batch process entire batch tensor (through block or directly)
        if self.use_block:
            # Move to device and process
            if embeddings_tensor.device != self.device:
                embeddings_tensor = embeddings_tensor.to(self.device)
            
            # Process entire batch tensor through block
            processed_tensor = self.model.block(embeddings_tensor)  # (batch_size, max_seq_len, hidden_size)
            processed_tensor = processed_tensor.cpu()
        else:
            # Don't use block, directly use pretrained model embeddings (move to CPU)
            processed_tensor = embeddings_tensor.cpu()
        
        # Infer hidden_size from processed tensor (if not using block)
        if not self.use_block:
            self.hidden_size = processed_tensor.shape[-1]
        
        # Return processed batch tensor and metadata (no merging)
        return processed_tensor, metadata_list


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate block output from pretrained LoRA model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--sequences", type=str, nargs="+", help="Input sequences")
    parser.add_argument("--output", type=str, help="Output save path")
    parser.add_argument("--batch_size", type=int, default=32, help="Inference batch size")
    parser.add_argument("--return_pooled", action="store_true", help="Return pooled output")
    
    args = parser.parse_args()
    
    # Initialize inference model
    inferencer = AdaptorInference(args.checkpoint)
    
    # Generate block output
    outputs = inferencer.generate_block_outputs(
        args.sequences,
        batch_size=args.batch_size,
        return_pooled=args.return_pooled
    )
    
    # Save output if path is provided
    if args.output:
        torch.save(outputs, args.output)
        print(f"✅ Output saved to {args.output}")
    else:
        print(f"📊 Generated output shape: {outputs.shape}")


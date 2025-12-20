"""
Model inference script for generating embeddings from pre-trained Mamba model.
This module imports the shared ModelInference class from main_pipeline.
"""
import sys
from pathlib import Path

# Add main_pipeline to path
main_pipeline_path = Path(__file__).parent.parent.parent / "main_pipeline"
sys.path.insert(0, str(main_pipeline_path))

# Import the shared ModelInference class from main_pipeline
# Use importlib to avoid circular import issues
import importlib.util
inference_module_path = main_pipeline_path / "inference.py"
spec = importlib.util.spec_from_file_location("main_pipeline_inference", inference_module_path)
main_pipeline_inference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(main_pipeline_inference)

# Get ModelInference class
ModelInference = main_pipeline_inference.ModelInference

# Re-export for backward compatibility
__all__ = ['ModelInference']

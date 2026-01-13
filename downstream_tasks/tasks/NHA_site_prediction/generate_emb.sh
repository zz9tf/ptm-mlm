#!/bin/bash
# Script to generate embeddings for NHA site prediction

# Check if pretrained_model_name is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <pretrained_model_name> [layer_index] [gpu_id]"
    echo "Examples:"
    echo "  $0 facebook/esm2_t30_150M_UR50D          # Use last layer, default GPU"
    echo "  $0 facebook/esm2_t30_150M_UR50D 12       # Use layer 12, default GPU"
    echo "  $0 facebook/esm2_t30_150M_UR50D 12 4     # Use layer 12, GPU 4"
    exit 1
fi

PRETRAINED_MODEL_NAME=$1
LAYER_INDEX_ARG=""
GPU_ID="4"  # Default GPU

# Check if layer_index is provided
if [ $# -ge 2 ]; then
    LAYER_INDEX=$2
    # 🔧 Handle "None" string - don't pass layer_index if it's None
    if [ "$LAYER_INDEX" != "None" ] && [ "$LAYER_INDEX" != "none" ] && [ -n "$LAYER_INDEX" ]; then
        LAYER_INDEX_ARG="--layer_index $LAYER_INDEX"
    fi
fi

# Check if gpu_id is provided
if [ $# -ge 3 ]; then
    GPU_ID=$3
fi

# Activate conda environment
source /home/zz/miniconda3/etc/profile.d/conda.sh
conda activate ptm

export CUDA_VISIBLE_DEVICES="$GPU_ID"
# Change to downstream_tasks directory (so embeddings go to downstream_tasks/embeddings)
cd /home/zz/zheng/ptm-mlm/downstream_tasks

echo "🚀 Generating embeddings for NHA site prediction..."
echo "Model: $PRETRAINED_MODEL_NAME"
if [ -n "$LAYER_INDEX_ARG" ]; then
    echo "Layer: $LAYER_INDEX"
elif [ $# -ge 2 ] && [ "$LAYER_INDEX" = "None" ]; then
    echo "Layer: None (using last layer)"
fi
echo "GPU: $GPU_ID"

# Run NHA site prediction embedding generation
python utils/embeddings_generator/nhas_generator.py --pretrained_model_name "$PRETRAINED_MODEL_NAME" $LAYER_INDEX_ARG

echo "✅ NHA site prediction embeddings generation completed!"
#!/bin/bash
# Script to generate embeddings for Phosphorylation site prediction

# Check if pretrained_model_name is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <pretrained_model_name> [layer_index]"
    echo "Examples:"
    echo "  $0 facebook/esm2_t30_150M_UR50D          # Use last layer"
    echo "  $0 facebook/esm2_t30_150M_UR50D 12       # Use layer 12 (0-based for ESM2, 1-based for ESMC)"
    exit 1
fi

PRETRAINED_MODEL_NAME=$1
LAYER_INDEX_ARG=""

# Check if layer_index is provided
if [ $# -ge 2 ]; then
    LAYER_INDEX=$2
    LAYER_INDEX_ARG="--layer_index $LAYER_INDEX"
fi

# Activate conda environment
source /home/zz/miniconda3/etc/profile.d/conda.sh
conda activate ptm

export CUDA_VISIBLE_DEVICES="2"
# Change to downstream_tasks directory (so embeddings go to downstream_tasks/embeddings)
cd /home/zz/zheng/ptm-mlm/downstream_tasks

echo "🚀 Generating embeddings for Phosphorylation site prediction..."
echo "Model: $PRETRAINED_MODEL_NAME"
if [ -n "$LAYER_INDEX_ARG" ]; then
    echo "Layer: $LAYER_INDEX"
fi

# Run Phosphorylation site prediction embedding generation
python utils/embeddings_generator/p_site_generator.py --pretrained_model_name "$PRETRAINED_MODEL_NAME" $LAYER_INDEX_ARG

echo "✅ Phosphorylation site prediction embeddings generation completed!"
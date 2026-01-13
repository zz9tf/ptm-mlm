#!/bin/bash
# Script to generate ESM embeddings using accelerate

source /home/zz/miniconda3/etc/profile.d/conda.sh
conda activate ptm
cd /home/zz/zheng/ptm-mlm/main_pipeline

# Exclude GPU 3, use GPUs 1,2,4,5 (will be mapped to 0,1,2,3 by CUDA_VISIBLE_DEVICES)
export CUDA_VISIBLE_DEVICES="0"

# Generate ESM embeddings for dataset
# Model and repr_layer are read from configs/base.yaml (training.esm_model and training.repr_layer)
# Dataset location is read from configs/base.yaml (preprocess.dataset_location)
# Original sequence column 'ori_seq' is hardcoded in the script
python generate_embeddings.py \
    --config configs/combined_emb.yaml \
    --output_dir datasets/embeddings/esm2_650m/32/memmap_combined \
    --batch_size 256 \
    --chunk_size_gb 10
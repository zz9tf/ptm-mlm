#!/bin/bash
# Script to generate ESM embeddings using accelerate

source /home/zz/miniconda3/etc/profile.d/conda.sh
conda activate ptm-mamba
cd /home/zz/zheng/ptm-mlm/main_pipeline

# Exclude GPU 3, use GPUs 0,1,2,4,5
export CUDA_VISIBLE_DEVICES="1,2,4,5"

# Generate embeddings with 4 GPUs
accelerate launch --num_processes=4 generate_embeddings.py \
    --config configs/base.yaml \
    --output_dir embeddings \
    --batch_size 6


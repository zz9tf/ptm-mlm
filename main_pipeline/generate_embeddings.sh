#!/bin/bash
# Script to generate ESM embeddings using accelerate

source /home/zz/miniconda3/etc/profile.d/conda.sh
conda activate ptm
cd /home/zz/zheng/ptm-mlm/main_pipeline

# Exclude GPU 3, use GPUs 1,2,4,5 (will be mapped to 0,1,2,3 by CUDA_VISIBLE_DEVICES)
export CUDA_VISIBLE_DEVICES="1,2,4,5"

# PTM MASKED sequences
# python generate_embeddings.py \
#     --config configs/base.yaml \
#     --output_dir embeddings \
#     --batch_size 6 \
#     --chunk_size_gb 10

# Original sequences
# 8 - 15b 64-esm_650m
accelerate launch --num_processes 4 generate_embeddings.py \
    --config configs/base.yaml \
    --use_original_sequence \
    --original_sequence_column ori_seq \
    --output_dir embeddings \
    --batch_size 64 \
    --chunk_size_gb 10 \
    --model esmc_300m # esm2_650m esm2_15b esm3_7b esmc_300m
    --repr_layer 26  # Optional: specify layer index (overrides config)

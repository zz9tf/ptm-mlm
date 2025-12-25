#!/bin/bash
# Script to generate ESM embeddings using accelerate

source /home/zz/miniconda3/etc/profile.d/conda.sh
conda activate ptm-mamba
cd /home/zz/zheng/ptm-mlm/main_pipeline

# Exclude GPU 3, use GPUs 1,2,4,5 (will be mapped to 0,1,2,3 by CUDA_VISIBLE_DEVICES)
export CUDA_VISIBLE_DEVICES="5"

# PTM MASKED sequences
# python generate_embeddings.py \
#     --config configs/base.yaml \
#     --output_dir embeddings \
#     --batch_size 6 \
#     --chunk_size_gb 10

# Original sequences
python generate_embeddings.py \
    --config configs/base.yaml \
    --use_original_sequence \
    --original_sequence_column ori_seq \
    --output_dir embeddings \
    --batch_size 8 \
    --chunk_size_gb 10

# Original sequences with ESM3 model
# python generate_embeddings.py \
#     --config configs/base.yaml \
#     --use_original_sequence \
#     --original_sequence_column ori_seq \
#     --output_dir embeddings \
#     --model esm3_7b \
#     --batch_size 8 \
#     --chunk_size_gb 100

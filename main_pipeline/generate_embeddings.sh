#!/bin/bash
# Script to generate ESM embeddings using accelerate

source /home/zz/miniconda3/etc/profile.d/conda.sh
conda activate ptm
cd /home/zz/zheng/ptm-mlm/main_pipeline

# Exclude GPU 3, use GPUs 1,2,4,5 (will be mapped to 0,1,2,3 by CUDA_VISIBLE_DEVICES)
export CUDA_VISIBLE_DEVICES="1"

# PTM MASKED sequences
# python generate_embeddings.py \
#     --config configs/base.yaml \
#     --output_dir embeddings \
#     --batch_size 6 \
#     --chunk_size_gb 10

# Original sequences
# Model and repr_layer are read from configs/base.yaml (training.esm_model and training.repr_layer)
# Dataset location is read from configs/base.yaml (preprocess.dataset_location)
python generate_embeddings.py \
    --config configs/base.yaml \
    --use_original_sequence \
    --original_sequence_column ori_seq \
    --output_dir embeddings_combined \
    --batch_size 512 \
    --chunk_size_gb 10 \
    --functional_role  # Extract functional_role labels from dataset CSV if it contains 'functional_role' and 'position' columns

#!/bin/bash
# Training script for only_mamba experiment

source /home/zz/miniconda3/etc/profile.d/conda.sh
conda activate ptm
cd /home/zz/zheng/ptm-mlm/main_pipeline

# Set GPU (use GPU 0)
export CUDA_VISIBLE_DEVICES="5"

# Run training with config that matches checkpoint
python train.py \
    --config configs/functional_resume.yaml
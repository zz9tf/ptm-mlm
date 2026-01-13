#!/bin/bash
# Training script for only_mamba experiment

source /home/zz/miniconda3/etc/profile.d/conda.sh
conda activate ptm
cd /home/zz/zheng/ptm-mlm/main_pipeline

# Set GPU (use GPU 0)
export CUDA_VISIBLE_DEVICES="4"

# Run training with config overrides
python train.py \
    --config configs/base.yaml \
    exp_name=combine \
    dataset.preload_all=true \
    dataset.dataset_dir=/home/zz/zheng/ptm-mlm/main_pipeline/memmap_functional \
    model.d_model=1024 \
    training.num_train_epochs=40 \
    training.per_device_train_batch_size=512
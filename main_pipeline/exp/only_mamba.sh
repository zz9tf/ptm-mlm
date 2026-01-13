#!/bin/bash
# Training script for only_mamba experiment

source /home/zz/miniconda3/etc/profile.d/conda.sh
conda activate ptm
cd /home/zz/zheng/ptm-mlm/main_pipeline

# Set GPU (use GPU 0)
export CUDA_VISIBLE_DEVICES="5"

# Run training with config overrides
python train.py \
    --config configs/base.yaml \
    exp_name=only_mamba \
    dataset.preload_all=true \
    dataset.dataset_dir=/home/zz/zheng/ptm-mlm/main_pipeline/memmap_mamba \
    model.d_model=1024 \
    training.num_train_epochs=1 \
    training.per_device_train_batch_size=512

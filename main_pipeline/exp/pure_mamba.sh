source /home/zz/miniconda3/etc/profile.d/conda.sh
conda activate ptm-mamba
cd /home/zz/zheng/ptm-mlm/main_pipeline

# Exclude GPU 3, use GPUs 0,1,2,4,5
export CUDA_VISIBLE_DEVICES="4,5"

# python train.py \
#     exp_name=pure_mamba_128 \
#     model.d_model=128 \
#     training.num_train_epochs=10000 \
#     training.per_device_train_batch_size=8192 \
#     training.resume_from_output=/home/zz/zheng/ptm-mlm/main_pipeline/outputs/pure_mamba_128-2025-12-22-15-01-34

python train.py \
    exp_name=pure_mamba_1024 \
    model.d_model=1024 \
    training.num_train_epochs=100 \
    training.per_device_train_batch_size=128

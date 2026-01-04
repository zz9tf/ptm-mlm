source /home/zz/miniconda3/etc/profile.d/conda.sh
conda activate ptm-mamba
cd /home/zz/zheng/ptm-mlm/main_pipeline

# Exclude GPU 3, use GPUs 0,1,2,4,5
export CUDA_VISIBLE_DEVICES="2"

python train.py \
    exp_name=esm_650m_based_mamba_512 \
    model.d_model=512 \
    model.esm_embed_dim=1280 \
    training.esm_model=esm2_650m \
    training.num_train_epochs=10 \
    training.per_device_train_batch_size=8 \
    training.use_esm=true \
    training.use_precomputed_embeddings=false
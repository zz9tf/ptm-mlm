#!/bin/bash
# Phosphorylation Site Prediction Pipeline
# 磷酸化位点预测完整流程脚本

source /home/zz/miniconda3/etc/profile.d/conda.sh
conda activate ptm-mamba
cd /home/zz/zheng/ptm-mlm/main_pipeline

set -e  # Exit on error
export CUDA_VISIBLE_DEVICES="0,1,2,4,5"

# ============================================
# 配置参数 (Configuration)
# ============================================
WORK_DIR="/home/zz/zheng/ptm-mlm/downstream_tasks/p_site_prediction"
CHECKPOINT="${WORK_DIR}/best.ckpt"  # 预训练模型checkpoint路径
TRAIN_DATA="${WORK_DIR}/PhosphositePTM.train.txt"
TEST_DATA="${WORK_DIR}/PhosphositePTM.test.txt"
VALID_DATA="${WORK_DIR}/PhosphositePTM.valid.txt"  # 可选
EMBEDDINGS_DIR="${WORK_DIR}/embeddings"
BATCH_SIZE=32
NUM_EPOCHS=10
LEARNING_RATE=1e-4
DEVICE="cuda"  # 或 "cpu"
LAMBDA_WEIGHT=0.5  # Weight (λ) for AUCMLoss in combined loss: loss = bce_loss + λ * auc_loss
MAX_SEQUENCE_LENGTH=512  # 滑动窗口大小（默认512，与训练配置一致）
WINDOW_OVERLAP=0.3  # 滑动窗口重叠比例（0.5表示50%重叠）
USE_SLIDING_WINDOW=true  # 使用滑动窗口处理长序列（推荐，确保所有位置都被处理）
MODEL_TYPE="esm2"  # 模型类型: "mamba" 或 "esm2"
ESM2_MODEL_NAME="facebook/esm2_t33_650M_UR50D"  # ESM2模型名称（仅在MODEL_TYPE="esm2"时使用）

cd "${WORK_DIR}"

# ============================================
# Step 1: 生成Embeddings (Generate Embeddings)
# ============================================
echo "============================================"
echo "Step 1: 生成Embeddings"
echo "============================================"

accelerate launch --num_processes=1 generate_embeddings.py \
    --model_type "${MODEL_TYPE}" \
    --checkpoint "${CHECKPOINT}" \
    --esm2_model_name "${ESM2_MODEL_NAME}" \
    --train_data "${TRAIN_DATA}" \
    --test_data "${TEST_DATA}" \
    --valid_data "${VALID_DATA}" \
    --output_dir "${EMBEDDINGS_DIR}" \
    --batch_size ${BATCH_SIZE} \
    --max_sequence_length ${MAX_SEQUENCE_LENGTH} \
    --window_overlap ${WINDOW_OVERLAP} \
    $([ "${USE_SLIDING_WINDOW}" = "true" ] && echo "--use_sliding_window" || echo "--no_sliding_window")

echo "✅ Step 1 完成: Embeddings已生成"
echo ""

# ============================================
# Step 2: 训练分类头并评估 (Train Classification Head and Evaluate)
# ============================================
echo "============================================"
echo "Step 2: 训练分类头并评估"
echo "============================================"

python train_and_evaluation.py \
    --embeddings_dir "${EMBEDDINGS_DIR}" \
    --num_epochs ${NUM_EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --output_dir "${WORK_DIR}" \
    --device "${DEVICE}" \
    --lambda_weight ${LAMBDA_WEIGHT}

echo "✅ Step 2 完成: 分类头训练完成，测试集评估完成"
echo ""

# echo "============================================"
# echo "🎉 完整流程执行完毕!"
# echo "============================================"
# echo "输出文件:"
# echo "  - Embeddings: ${EMBEDDINGS_DIR}/"
# echo "  - 训练好的模型: ${WORK_DIR}/trained_head.pt (包含训练/验证/测试集指标)"
# echo "============================================"


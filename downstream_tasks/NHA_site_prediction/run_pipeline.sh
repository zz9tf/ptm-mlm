#!/bin/bash
# NHA Site Prediction Pipeline
# NHA位点预测完整流程脚本

set -e  # Exit on error
export CUDA_VISIBLE_DEVICES="1,2,4,5"

# ============================================
# 配置参数 (Configuration)
# ============================================
WORK_DIR="/home/zz/zheng/ptm-mlm/downstream_tasks/NHA_site_prediction"
CHECKPOINT="${WORK_DIR}/../best.ckpt"  # 预训练模型checkpoint路径
DATA="${WORK_DIR}/NHAC.csv"  # NHA数据文件
EMBEDDINGS_DIR="${WORK_DIR}/embeddings"
BATCH_SIZE=32
NUM_EPOCHS=10
LEARNING_RATE=1e-4
DEVICE="cuda"  # 或 "cpu"
LAMBDA_WEIGHT=0.1  # Weight (λ) for AUCMLoss in combined loss: loss = bce_loss + λ * auc_loss
MAX_SEQUENCE_LENGTH=512  # 最大序列长度（seq_61只有61，所以不需要滑动窗口）
MODEL_TYPE="esm2"  # 模型类型: "mamba" 或 "esm2"
ESM2_MODEL_NAME="facebook/esm2_t33_650M_UR50D"  # ESM2模型名称（仅在MODEL_TYPE="esm2"时使用）

cd "${WORK_DIR}"

# ============================================
# Step 1: 生成Embeddings (Generate Embeddings)
# ============================================
echo "============================================"
echo "Step 1: 生成Embeddings"
echo "============================================"

python3 generate_embeddings.py \
    --model_type "${MODEL_TYPE}" \
    --checkpoint "${CHECKPOINT}" \
    --esm2_model_name "${ESM2_MODEL_NAME}" \
    --data "${DATA}" \
    --output_dir "${EMBEDDINGS_DIR}" \
    --batch_size ${BATCH_SIZE} \
    --max_sequence_length ${MAX_SEQUENCE_LENGTH}

echo "✅ Step 1 完成: Embeddings已生成"
echo ""

# ============================================
# Step 2: 训练分类头并评估 (Train Classification Head and Evaluate)
# ============================================
echo "============================================"
echo "Step 2: 训练分类头并评估"
echo "============================================"

python3 train_and_evaluation.py \
    --embeddings_dir "${EMBEDDINGS_DIR}" \
    --num_epochs ${NUM_EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --output_dir "${WORK_DIR}" \
    --device "${DEVICE}" \
    --lambda_weight ${LAMBDA_WEIGHT}

echo "✅ Step 2 完成: 分类头训练完成，测试集评估完成"
echo ""

# ============================================
# 🎉 完整流程执行完毕!
# ============================================
echo "============================================"
echo "🎉 完整流程执行完毕!"
echo "============================================"
echo "输出文件:"
echo "  - Embeddings: ${EMBEDDINGS_DIR}/"
echo "  - 训练好的模型 (包含训练/验证/测试集指标): ${WORK_DIR}/trained_head.pt"
echo "============================================"


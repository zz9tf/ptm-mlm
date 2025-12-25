#!/bin/bash
# NHA Site Prediction Pipeline
# NHA位点预测完整流程脚本

source /home/zz/miniconda3/etc/profile.d/conda.sh
conda activate ptm-mamba
cd /home/zz/zheng/ptm-mlm/main_pipeline

set -e  # Exit on error
export CUDA_VISIBLE_DEVICES="4"

# ============================================
# 配置参数 (Configuration)
# ============================================
WORK_DIR="/home/zz/zheng/ptm-mlm/downstream_tasks/NHA_site_prediction"
CHECKPOINT="${WORK_DIR}/../checkpoints/1024.ckpt"  # 预训练模型checkpoint路径
DATA="${WORK_DIR}/NHAC.csv"  # NHA数据文件
BASE_OUTPUT_DIR="/home/zz/zheng/ptm-mlm/downstream_tasks/outputs"  # 基础输出目录
# 创建带日期的输出目录
DATE_STR=$(date +"%Y-%m-%d")
OUTPUT_DIR="${BASE_OUTPUT_DIR}/NHA_site_prediction_${DATE_STR}"
BATCH_SIZE=320
NUM_EPOCHS=10
LEARNING_RATE=1e-4
DEVICE="cuda"  # 或 "cpu"
LAMBDA_WEIGHT=0.1  # Weight (λ) for AUCMLoss in combined loss: loss = bce_loss + λ * auc_loss
MAX_SEQUENCE_LENGTH=512  # 最大序列长度（seq_61只有61，所以不需要滑动窗口）
MODEL_TYPE="mamba"  # 模型类型: "mamba" 或 "esm2"
ESM2_MODEL_NAME="facebook/esm2_t33_650M_UR50D"  # ESM2模型名称（仅在MODEL_TYPE="esm2"时使用）
USE_ESM=false  # 是否加载ESM2-15B模型（仅在MODEL_TYPE="mamba"时有效）
                # true: 使用Mamba+ESM2组合（匹配训练时的配置）
                # false: 只使用Mamba模型（不加载ESM，节省内存）

# 创建输出目录
mkdir -p "${OUTPUT_DIR}"

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
    --output_dir "${OUTPUT_DIR}" \
    --batch_size ${BATCH_SIZE} \
    --max_sequence_length ${MAX_SEQUENCE_LENGTH} \
    $([ "${USE_ESM}" = "true" ] && echo "--use_esm" || echo "")

echo "✅ Step 1 完成: Embeddings已生成"
echo ""

# ============================================
# Step 2: 训练分类头并评估 (Train Classification Head and Evaluate)
# ============================================
echo "============================================"
echo "Step 2: 训练分类头并评估"
echo "============================================"

python3 train_and_evaluation.py \
    --output_dir "${OUTPUT_DIR}" \
    --num_epochs ${NUM_EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
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
echo "  - Embeddings: ${OUTPUT_DIR}/embeddings/ (自动存储在outputs目录下)"
echo "  - 训练好的模型 (包含训练/验证/测试集指标): ${OUTPUT_DIR}/trained_head.pt"
echo "  - 输出目录: ${OUTPUT_DIR}/"
echo "============================================"


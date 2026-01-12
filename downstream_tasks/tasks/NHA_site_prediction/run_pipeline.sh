#!/bin/bash
# NHA Site Prediction Pipeline
# NHA位点预测完整流程脚本

source /home/zz/miniconda3/etc/profile.d/conda.sh
conda activate ptm
cd /home/zz/zheng/ptm-mlm/main_pipeline

set -e  # Exit on error
export CUDA_VISIBLE_DEVICES="1"

# ============================================
# 配置参数 (Configuration)
# ============================================
WORK_DIR="/home/zz/zheng/ptm-mlm/downstream_tasks/NHA_site_prediction"
CHECKPOINT="/home/zz/zheng/ptm-mlm/downstream_tasks/checkpoints/LoRA_combine_ptm.ckpt"  # LoRA模型checkpoint路径
DATA="${WORK_DIR}/NHAC.csv"  # NHA数据文件
BASE_OUTPUT_DIR="/home/zz/zheng/ptm-mlm/downstream_tasks/outputs"  # 基础输出目录
# 创建带日期的输出目录
DATE_STR=$(date +"%Y-%m-%d")
OUTPUT_DIR="${BASE_OUTPUT_DIR}/NHA_site_prediction_lora_combine_ptm_${DATE_STR}"
BATCH_SIZE=512
NUM_EPOCHS=10
LEARNING_RATE=1e-4
DEVICE="cuda"  # 或 "cpu"
LAMBDA_WEIGHT=0.1  # Weight (λ) for AUCMLoss in combined loss: loss = bce_loss + λ * auc_loss
MAX_SEQUENCE_LENGTH=512  # 最大序列长度（seq_61只有61，所以不需要滑动窗口）
MODEL_TYPE="esmc"  # 模型类型: "mamba", "esm2", "lora", 或 "esmc"
USE_ESM=true  # 是否加载ESM2-15B模型（仅在MODEL_TYPE="mamba"时有效）
                # true: 使用Mamba+ESM2-15B组合（自动加载esm2_t48_15B_UR50D，匹配训练时的配置）
                # false: 只使用Mamba模型（不加载ESM，节省内存）
TRAIN_BY_LENGTH_GROUPS=true  # 是否按长度分组训练
                              # true: 为每个序列长度训练独立的模型（确保训练稳定，位置信息明确）
                              # false: 训练单个模型处理所有长度（使用LengthGroupedBatchSampler）

# 创建输出目录
mkdir -p "${OUTPUT_DIR}"

cd "${WORK_DIR}"

# # ============================================
# # Step 1: 生成Embeddings (Generate Embeddings)
# # ============================================
# echo "============================================"
# echo "Step 1: 生成Embeddings"
# echo "============================================"

# python3 generate_embeddings.py \
#     --model_type "${MODEL_TYPE}" \
#     --data "${DATA}" \
#     --output_dir "${OUTPUT_DIR}" \
#     --batch_size ${BATCH_SIZE} \
#     --max_sequence_length ${MAX_SEQUENCE_LENGTH}

# echo "✅ Step 1 完成: Embeddings已生成"
# echo ""

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
    --lambda_weight ${LAMBDA_WEIGHT} \
    $([ "${TRAIN_BY_LENGTH_GROUPS}" = "true" ] && echo "--train_by_length_groups" || echo "")

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
if [ "${TRAIN_BY_LENGTH_GROUPS}" = "true" ]; then
    echo "  - 按长度分组的模型: ${OUTPUT_DIR}/length_*/ (每个长度一个独立模型)"
    echo "  - 长度分组摘要: ${OUTPUT_DIR}/length_groups_summary.json"
else
    echo "  - 训练好的模型 (包含训练/验证/测试集指标): ${OUTPUT_DIR}/trained_head.pt"
fi
echo "  - 输出目录: ${OUTPUT_DIR}/"
echo "============================================"


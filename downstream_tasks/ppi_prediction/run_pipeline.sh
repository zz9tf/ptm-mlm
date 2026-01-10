#!/bin/bash
# PPI Prediction Pipeline
# 蛋白质-蛋白质相互作用预测完整流程脚本

source /home/zz/miniconda3/etc/profile.d/conda.sh
conda activate ptm
cd /home/zz/zheng/ptm-mlm/main_pipeline

set -e  # Exit on error
export CUDA_VISIBLE_DEVICES="4"

# ============================================
# 配置参数 (Configuration)
# ============================================
WORK_DIR="/home/zz/zheng/ptm-mlm/downstream_tasks/ppi_prediction"
CHECKPOINT="${WORK_DIR}/../checkpoints/LoRA_ptm.ckpt"  # LoRA模型checkpoint路径
DATA="${WORK_DIR}/PTM experimental evidence.csv"  # PPI数据文件
BASE_OUTPUT_DIR="/home/zz/zheng/ptm-mlm/downstream_tasks/outputs"  # 基础输出目录
# 创建带日期的输出目录
DATE_STR=$(date +"%Y-%m-%d")
OUTPUT_DIR="${BASE_OUTPUT_DIR}/ppi_prediction_lora_ptm_${DATE_STR}"
BATCH_SIZE=512
NUM_EPOCHS=50
LEARNING_RATE=1e-4
DEVICE="cuda"  # 或 "cpu"
MAX_SEQUENCE_LENGTH=512  # 最大序列长度
MAX_LENGTH=2000  # TransformerClassifier的最大长度参数
DROPOUT=0.3  # Dropout rate
MODEL_TYPE="esmc"  # 模型类型: "mamba", "esm2", "lora", 或 "esmc"
USE_ESM=true  # 是否加载ESM2-15B模型（仅在MODEL_TYPE="mamba"时有效）
                # true: 使用Mamba+ESM2-15B组合（自动加载esm2_t48_15B_UR50D，匹配训练时的配置）
                # false: 只使用Mamba模型（不加载ESM，节省内存）
TRAIN_RATIO=0.7  # 训练集比例
VALID_RATIO=0.15  # 验证集比例
TEST_RATIO=0.15  # 测试集比例
RANDOM_SEED=42  # 随机种子

# 创建输出目录
mkdir -p "${OUTPUT_DIR}"

cd "${WORK_DIR}"

# # ============================================
# # Step 1: 生成Embeddings (Generate Embeddings)
# # ============================================
# echo "============================================"
# echo "Step 1: 生成Embeddings (Binder, WT, PTM)"
# echo "============================================"

# python3 generate_embeddings.py \
#     --model_type "${MODEL_TYPE}" \
#     --data "${DATA}" \
#     --output_dir "${OUTPUT_DIR}" \
#     --batch_size ${BATCH_SIZE} \
#     --max_sequence_length ${MAX_SEQUENCE_LENGTH} \
#     --train_ratio ${TRAIN_RATIO} \
#     --valid_ratio ${VALID_RATIO} \
#     --test_ratio ${TEST_RATIO} \
#     --random_seed ${RANDOM_SEED}

# echo "✅ Step 1 完成: Embeddings已生成"
# echo ""

# ============================================
# Step 2: 训练分类头并评估 (Train Classification Head and Evaluate)
# ============================================
echo "============================================"
echo "Step 2: 训练TransformerClassifier并评估"
echo "============================================"

python3 train_and_evaluation.py \
    --output_dir "${OUTPUT_DIR}" \
    --num_epochs ${NUM_EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --dropout ${DROPOUT} \
    --max_length ${MAX_LENGTH} \
    --device "${DEVICE}"

echo "✅ Step 2 完成: 分类头训练完成，测试集评估完成"
echo ""

# ============================================
# 🎉 完整流程执行完毕!
# ============================================
echo "============================================"
echo "🎉 完整流程执行完毕!"
echo "============================================"
echo "输出文件:"
echo "  - Embeddings: ${OUTPUT_DIR}/embeddings/"
echo "    * train_binder_embeddings.pt, train_wt_embeddings.pt, train_ptm_embeddings.pt"
echo "    * valid_binder_embeddings.pt, valid_wt_embeddings.pt, valid_ptm_embeddings.pt"
echo "    * test_binder_embeddings.pt, test_wt_embeddings.pt, test_ptm_embeddings.pt"
echo "    * train_labels.pt, valid_labels.pt, test_labels.pt"
echo "  - 训练好的模型: ${OUTPUT_DIR}/best_model.pt"
echo "  - 训练历史: ${OUTPUT_DIR}/training_history.json"
echo "  - 训练曲线: ${OUTPUT_DIR}/training_curves.png"
echo "  - 测试指标: ${OUTPUT_DIR}/test_metrics.json"
echo "  - 输出目录: ${OUTPUT_DIR}/"
echo "============================================"


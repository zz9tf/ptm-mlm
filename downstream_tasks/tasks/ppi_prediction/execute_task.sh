#!/bin/bash
# PPI Prediction Pipeline
# 蛋白质-蛋白质相互作用预测完整流程脚本

# 检查参数
if [ $# -ne 4 ]; then
    echo "Usage: $0 <adaptor_checkpoint> <model_name> <layer> <gpu_device>"
    echo "Example: $0 LoRA_combine EvolutionaryScale_esmc-600m-2024-12 30 4"
    echo "Example (no checkpoint): $0 None EvolutionaryScale_esmc-600m-2024-12 30 4"
    echo "Example (no checkpoint): $0 \"\" EvolutionaryScale_esmc-600m-2024-12 30 4"
    exit 1
fi

ADAPTOR_CHECKPOINT="$1"
# 处理空字符串或 "None" 字符串，转换为真正的 None
if [ -z "$ADAPTOR_CHECKPOINT" ] || [ "$ADAPTOR_CHECKPOINT" = "None" ] || [ "$ADAPTOR_CHECKPOINT" = "none" ]; then
    ADAPTOR_CHECKPOINT="None"
fi
MODEL_NAME="$2"
LAYER_INDEX="$3"
GPU_DEVICE="$4"

# 激活环境
source /home/zz/miniconda3/etc/profile.d/conda.sh
conda activate ptm

set -e  # Exit on error
export CUDA_VISIBLE_DEVICES="${GPU_DEVICE}"

# ============================================
# 配置参数 (Configuration)
# ============================================
WORK_DIR="/home/zz/zheng/ptm-mlm/downstream_tasks/tasks/ppi_prediction"
BASE_OUTPUT_DIR="/home/zz/zheng/ptm-mlm/downstream_tasks/outputs"

# 创建带日期的输出目录
DATE_STR=$(date +"%Y-%m-%d")
OUTPUT_DIR="${BASE_OUTPUT_DIR}/ppi_prediction_${ADAPTOR_CHECKPOINT}_${DATE_STR}"

# 训练参数
BATCH_SIZE=512
NUM_EPOCHS=50
LEARNING_RATE=1e-4
DEVICE="cuda"
DROPOUT=0.3
MAX_LENGTH=2000

echo "🔄 Starting PPI prediction pipeline..."
if [ -z "$ADAPTOR_CHECKPOINT" ] || [ "$ADAPTOR_CHECKPOINT" = "None" ] || [ "$ADAPTOR_CHECKPOINT" = "none" ]; then
    echo "   Adaptor checkpoint: None (using raw embeddings)"
else
    echo "   Adaptor checkpoint: ${ADAPTOR_CHECKPOINT}"
fi
echo "   Model name: ${MODEL_NAME}"
echo "   Layer index: ${LAYER_INDEX}"
echo "   GPU device: ${GPU_DEVICE}"
echo "   Output dir: ${OUTPUT_DIR}"

# 创建输出目录
mkdir -p "${OUTPUT_DIR}"

cd "${WORK_DIR}"

# 🔧 构建 Python 命令参数
PYTHON_ARGS=(
    --output_dir "${OUTPUT_DIR}"
    --num_epochs ${NUM_EPOCHS}
    --batch_size ${BATCH_SIZE}
    --learning_rate ${LEARNING_RATE}
    --dropout ${DROPOUT}
    --max_length ${MAX_LENGTH}
    --device "${DEVICE}"
)

# 🔧 model_name 必须传递（不能为空）
if [ -z "$MODEL_NAME" ]; then
    echo "❌ 错误: model_name 不能为空!"
    exit 1
fi
PYTHON_ARGS+=(--model_name "${MODEL_NAME}")

# 🔧 只有当 layer_index 不为空且不是 "None" 时才添加（因为 Python 期望 int 类型）
if [ -n "$LAYER_INDEX" ] && [ "$LAYER_INDEX" != "None" ] && [ "$LAYER_INDEX" != "none" ]; then
    PYTHON_ARGS+=(--layer_index ${LAYER_INDEX})
fi

# 🔧 只有当 adaptor_checkpoint 不为空且不是 "None" 时才添加
if [ -n "$ADAPTOR_CHECKPOINT" ] && [ "$ADAPTOR_CHECKPOINT" != "None" ] && [ "$ADAPTOR_CHECKPOINT" != "none" ]; then
    PYTHON_ARGS+=(--adaptor_checkpoint "${ADAPTOR_CHECKPOINT}")
fi

python3 train_and_evaluation.py "${PYTHON_ARGS[@]}"

echo "✅ Step 2 完成: TransformerClassifier训练完成，测试集评估完成"
echo ""

# ============================================
# 🎉 完整流程执行完毕!
# ============================================
echo "============================================"
echo "🎉 完整流程执行完毕!"
echo "============================================"
echo "输出文件:"
echo "  - 训练好的模型: ${OUTPUT_DIR}/best_model.pt"
echo "  - 训练历史: ${OUTPUT_DIR}/training_history.json"
echo "  - 训练曲线: ${OUTPUT_DIR}/training_curves.png"
echo "  - 测试指标: ${OUTPUT_DIR}/test_metrics.json"
echo "  - 输出目录: ${OUTPUT_DIR}/"
echo "============================================"


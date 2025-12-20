# 磷酸化位点预测 - 运行命令指南

## 快速开始 (Quick Start)

### 方式1: 一键运行完整流程
```bash
cd /home/zz/zheng/ptm-mlm/downstream_tasks/p_site_prediction
chmod +x run_pipeline.sh
./run_pipeline.sh
```

---

## 分步执行 (Step by Step)

### Step 1: 生成Embeddings

使用预训练模型为训练集和测试集生成embeddings：

```bash
cd /home/zz/zheng/ptm-mlm/downstream_tasks/p_site_prediction

python generate_embeddings.py \
    --checkpoint /home/zz/zheng/ptm-mlm/downstream_tasks/p_site_prediction/best.ckpt \
    --train_data /home/zz/zheng/ptm-mlm/downstream_tasks/p_site_prediction/PhosphositePTM.train.txt \
    --test_data /home/zz/zheng/ptm-mlm/downstream_tasks/p_site_prediction/PhosphositePTM.test.txt \
    --output_dir /home/zz/zheng/ptm-mlm/downstream_tasks/p_site_prediction/embeddings \
    --batch_size 32 \
    --max_sequence_length 512 \
    --use_sliding_window \
    --window_overlap 0.5
```

**参数说明:**
- `--checkpoint`: 预训练模型checkpoint路径
- `--train_data`: 训练数据CSV文件路径 (包含seq和label列)
- `--test_data`: 测试数据CSV文件路径
- `--output_dir`: Embeddings输出目录
- `--batch_size`: 批处理大小 (默认: 32)
- `--max_sequence_length`: 滑动窗口大小 (默认: 512，与训练配置一致)
  - 对于超过此长度的序列，会使用滑动窗口处理
- `--use_sliding_window`: 使用滑动窗口处理长序列 (默认: True，强烈推荐)
  - ✅ **推荐**: 确保所有序列位置都被处理，保留完整的标签信息
  - 对于位点预测任务，这是必需的，否则无法计算完整的准确率
- `--window_overlap`: 滑动窗口重叠比例 (默认: 0.5，即50%重叠)
  - 范围: 0.0 到 1.0
  - 更高的重叠比例提供更好的上下文，但需要更多计算
  - 0.5 (50%重叠) 是平衡性能和准确性的好选择

**输出文件:**
- `embeddings/train_embeddings.pt` - 训练集embeddings
- `embeddings/train_labels.pt` - 训练集标签
- `embeddings/train_sequences.pt` - 训练集序列
- `embeddings/test_embeddings.pt` - 测试集embeddings
- `embeddings/test_labels.pt` - 测试集标签
- `embeddings/test_sequences.pt` - 测试集序列

---

### Step 2: 训练分类头

在生成的embeddings上训练分类头模型：

```bash
python train_head.py \
    --embeddings_dir /home/zz/zheng/ptm-mlm/downstream_tasks/p_site_prediction/embeddings \
    --num_epochs 10 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --dropout 0.1 \
    --output_dir /home/zz/zheng/ptm-mlm/downstream_tasks/p_site_prediction \
    --device cuda
```

**参数说明:**
- `--embeddings_dir`: Embeddings目录路径
- `--num_epochs`: 训练轮数 (默认: 10)
- `--batch_size`: 批处理大小 (默认: 32)
- `--learning_rate`: 学习率 (默认: 1e-4)
- `--dropout`: Dropout率 (默认: 0.1)
- `--output_dir`: 模型保存目录
- `--device`: 设备 (cuda/cpu, 默认: 自动检测)

**输出文件:**
- `trained_head.pt` - 训练好的分类头模型

---

### Step 3: 预测

使用训练好的模型进行预测：

```bash
python predict.py \
    --embeddings_dir /home/zz/zheng/ptm-mlm/downstream_tasks/p_site_prediction/embeddings \
    --model_path /home/zz/zheng/ptm-mlm/downstream_tasks/p_site_prediction/trained_head.pt \
    --batch_size 32 \
    --output_path /home/zz/zheng/ptm-mlm/downstream_tasks/p_site_prediction/test_predictions.txt \
    --device cuda \
    --evaluate
```

**参数说明:**
- `--embeddings_dir`: Embeddings目录路径
- `--model_path`: 训练好的模型路径
- `--batch_size`: 批处理大小 (默认: 32)
- `--output_path`: 预测结果输出路径
- `--device`: 设备 (cuda/cpu)
- `--evaluate`: 如果提供此标志，会计算评估指标 (需要test_labels.pt)

**输出文件:**
- `test_predictions.txt` - 预测结果CSV文件 (包含seq和prediction列)

---

## 使用自定义路径

如果使用不同的路径，可以修改参数：

```bash
# Step 1: 生成embeddings
python generate_embeddings.py \
    --checkpoint /path/to/your/checkpoint.ckpt \
    --train_data /path/to/train.txt \
    --test_data /path/to/test.txt \
    --output_dir /path/to/embeddings \
    --batch_size 64

# Step 2: 训练
python train_head.py \
    --embeddings_dir /path/to/embeddings \
    --num_epochs 20 \
    --batch_size 64 \
    --learning_rate 5e-5 \
    --output_dir /path/to/output \
    --device cuda

# Step 3: 预测
python predict.py \
    --embeddings_dir /path/to/embeddings \
    --model_path /path/to/output/trained_head.pt \
    --output_path /path/to/predictions.txt \
    --device cuda \
    --evaluate
```

---

## 注意事项

1. **Checkpoint路径**: 确保checkpoint文件存在且路径正确
2. **数据格式**: CSV文件必须包含`seq`和`label`两列
3. **GPU内存**: 如果GPU内存不足，可以减小`batch_size`
4. **序列长度和滑动窗口**: 
   - 默认窗口大小为512（与训练配置一致）
   - **滑动窗口已默认启用**，确保所有序列位置都被处理
   - 对于长序列（>512），会自动使用滑动窗口分割处理
   - 重叠区域会取平均值，确保平滑过渡
   - 这样可以：
     - ✅ 保留完整的标签信息
     - ✅ 计算完整的准确率等指标
     - ✅ 对所有位置进行预测
5. **验证集**: 代码已支持处理valid集，使用`--valid_data`参数

---

## 文件结构

```
p_site_prediction/
├── generate_embeddings.py    # Step 1: 生成embeddings
├── train_head.py             # Step 2: 训练分类头
├── predict.py                # Step 3: 预测
├── prediction_head.py        # 分类头模型定义
├── inference.py              # 模型推理工具类
├── PhosphositePTM.train.txt  # 训练数据
├── PhosphositePTM.test.txt   # 测试数据
├── PhosphositePTM.valid.txt # 验证数据
├── run_pipeline.sh           # 一键运行脚本
└── embeddings/               # Embeddings输出目录
    ├── train_embeddings.pt
    ├── train_labels.pt
    ├── train_sequences.pt
    ├── test_embeddings.pt
    ├── test_labels.pt
    └── test_sequences.pt
```


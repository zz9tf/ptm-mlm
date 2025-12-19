# PTM-Mamba 训练 Pipeline

## 完整流程概览

```
1. 生成 ESM Embeddings (可选，但推荐)
   ↓
2. 训练模型
   ↓
3. 查看结果
```

## 详细步骤

### 步骤 1: 预生成 ESM Embeddings (推荐)

**目的**: 提前计算所有序列的 ESM embeddings，加速训练过程

**运行方式**:

```bash
# 单 GPU
python generate_embeddings.py \
    --config configs/base.yaml \
    --output_dir embeddings \
    --batch_size 8

# 多 GPU (推荐，更快)
accelerate launch --num_processes=4 generate_embeddings.py \
    --config configs/base.yaml \
    --output_dir embeddings \
    --batch_size 8
```

**输出**:
- `embeddings/train_embeddings.h5`
- `embeddings/val_embeddings.h5`
- `embeddings/test_embeddings.h5`

**时间**: 取决于数据集大小，使用多 GPU 可以显著加速

---

### 步骤 2: 训练模型

**目的**: 训练 PTM-Mamba 模型

**配置准备**:

编辑 `configs/base.yaml`，如果使用了预生成的 embeddings，设置：

```yaml
training:
  use_esm: true
  use_precomputed_embeddings: true  # 如果使用预生成的 embeddings
  embeddings_dir: "embeddings"      # embeddings 目录路径
```

**运行方式**:

```bash
# 使用启动脚本 (推荐)
bash exp/base_mamba.sh

# 或直接使用 accelerate
accelerate launch --num_processes=4 train.py --config configs/base.yaml
```

**输出目录结构**:
```
outputs/ptm-mamba-YYYY-MM-DD-HH-MM-SS/
├── config.yaml              # 训练配置
├── split_mapping.json       # 数据集划分映射 (unique_id -> split)
├── metrics.json             # 训练指标
├── best.ckpt                # 最佳模型
└── last.ckpt                # 最后一个 epoch 的模型
```

---

### 步骤 3: 查看结果

**指标文件**: `outputs/ptm-mamba-*/metrics.json`

包含：
- 训练指标 (每个 step)
- 验证指标 (每个 batch + 平均值)
- 测试指标 (每个 batch + 平均值)
- 数据集划分信息

**模型检查点**:
- `best.ckpt`: 验证集上表现最好的模型
- `last.ckpt`: 最后一个 epoch 的模型

---

## 完整 Pipeline 示例

### 方式 1: 使用预生成 Embeddings (推荐，更快)

```bash
# 1. 生成 embeddings (一次性，可以提前做)
accelerate launch --num_processes=4 generate_embeddings.py \
    --config configs/base.yaml \
    --output_dir embeddings \
    --batch_size 8

# 2. 配置训练使用预生成的 embeddings
# 编辑 configs/base.yaml:
# training:
#   use_precomputed_embeddings: true
#   embeddings_dir: "embeddings"

# 3. 开始训练
bash exp/base_mamba.sh
```

### 方式 2: 实时计算 Embeddings (更慢但不需要预处理)

```bash
# 1. 配置训练使用实时计算
# 编辑 configs/base.yaml:
# training:
#   use_esm: true
#   use_precomputed_embeddings: false

# 2. 直接开始训练
bash exp/base_mamba.sh
```

---

## 配置文件说明

### `configs/base.yaml` 关键参数

```yaml
# 数据集配置
dataset:
  dataset_location: "datasets/mamba.csv"
  sequence_column_name: "ptm_seq"
  val_size: 10000
  test_size: 5000
  split_seed: 42

# 训练配置
training:
  use_esm: true                          # 是否使用 ESM embeddings
  use_precomputed_embeddings: false      # 是否使用预生成的 embeddings
  embeddings_dir: null                   # embeddings 目录 (如果使用预生成的)
  num_train_epochs: 10                   # 训练轮数（总步数 = epochs × batches_per_epoch）
  log_steps: 100                         # 验证频率：每 N 步进行一次验证
  per_device_train_batch_size: 8        # 每个 GPU 的 batch size
  max_tokens_per_batch: 100000           # 每个 batch 的最大 token 数
  lr: 4e-4                               # 学习率
```

---

## 常见问题

### Q: 如何选择使用预生成还是实时计算 embeddings？

**A**: 
- **预生成**: 适合多次训练、调试，或数据集较大时
- **实时计算**: 适合单次训练、快速实验，或存储空间有限时

### Q: 如何恢复训练？

**A**: 在 `configs/base.yaml` 中设置：
```yaml
training:
  resume_from_checkpoint: "outputs/ptm-mamba-YYYY-MM-DD-HH-MM-SS/last.ckpt"
```

### Q: 如何控制训练步数？

**A**: 
- **训练总步数**: 由 `num_train_epochs` 和数据集大小决定（总步数 = epochs × batches_per_epoch）
- **验证频率**: `log_steps` 控制每 N 步进行一次验证（例如 `log_steps: 100` 表示每 100 步验证一次）

如果要训练固定步数，可以：
1. 调整 `num_train_epochs` 来控制训练轮数
2. 或者修改代码逻辑，添加 `max_steps` 参数来限制总步数

### Q: 多 GPU 训练如何配置？

**A**: 
- 使用 `CUDA_VISIBLE_DEVICES` 指定 GPU
- 使用 `accelerate launch --num_processes=N` 指定进程数
- 参考 `exp/base_mamba.sh` 中的配置

---

## 输出文件说明

### `split_mapping.json`
记录每个样本 (通过 unique_id) 属于哪个数据集 (train/val/test)，用于追踪和复现。

### `metrics.json`
包含所有训练、验证、测试指标，格式：
```json
[
  {"train_loss": 2.5, "train_acc": 0.3, ...},
  {"Epoch": 0, "Step": 100, "avg_val_loss": 2.1, ...},
  ...
]
```

### `config.yaml`
训练时使用的完整配置，用于复现实验。

---

## 性能优化建议

1. **使用预生成 embeddings**: 可以节省大量训练时间
2. **多 GPU 训练**: 使用 `accelerate launch` 加速
3. **调整 batch size**: 根据 GPU 内存调整 `per_device_train_batch_size`
4. **调整 max_tokens_per_batch**: 控制每个 batch 的大小


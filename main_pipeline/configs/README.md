# Configuration Guide

## 基础配置

使用 `configs/base.yaml` 作为基础配置文件，包含所有默认设置。

## 使用方法

### 1. 使用默认配置（base.yaml）

```bash
python main.py
```

### 2. 使用不同的配置文件

如果将来需要创建新的配置文件（例如 `base_v2.yaml`），可以通过 `--config-name` 指定：

```bash
python main.py --config-name=base_v2
```

### 3. 命令行参数覆盖（推荐用于 ablation study）

直接在命令行覆盖配置值，无需创建新文件：

```bash
# Ablation: 启用 ESM
python main.py training.use_esm=true

# Ablation: 大模型
python main.py model.d_model=1024 model.n_layer=24

# Ablation: 组合多个参数
python main.py training.use_esm=true model.d_model=1024 training.lr=2e-4

# Ablation: 修改数据集
python main.py dataset.dataset_loc=/path/to/other.csv dataset.val_size=5000
```

## 常用 Ablation Study 命令示例

```bash
# 实验1: 无ESM，小模型
python main.py training.use_esm=false model.d_model=256 model.n_layer=6

# 实验2: 有ESM，小模型
python main.py training.use_esm=true model.d_model=256 model.n_layer=6

# 实验3: 无ESM，大模型
python main.py training.use_esm=false model.d_model=1024 model.n_layer=24 training.lr=2e-4

# 实验4: 有ESM，大模型
python main.py training.use_esm=true model.d_model=1024 model.n_layer=24 training.lr=2e-4

# 修改学习率
python main.py training.lr=1e-4

# 修改批次大小
python main.py training.per_device_train_batch_size=16

# 修改保存路径
python main.py training.save_dir=./checkpoints/my_experiment
```

## 配置项说明

### dataset
- `dataset_loc`: 数据集文件路径
- `sequence_column_name`: 序列列名
- `val_size`: 验证集大小
- `test_size`: 测试集大小
- `subsample_size`: 子采样大小（null=全部）
- `max_sequence_length`: 最大序列长度（null=无限制）

### training
- `use_esm`: 是否使用ESM embeddings
- `num_train_epochs`: 训练轮数
- `lr`: 学习率
- `per_device_train_batch_size`: 每设备批次大小
- `max_tokens_per_batch`: 每批次最大token数
- `save_dir`: 模型保存目录

### model
- `d_model`: 模型维度
- `n_layer`: 层数
- `vocab_size`: 词汇表大小（自动设置）

## 多GPU训练

```bash
# 使用 accelerate
accelerate launch --num_processes=4 main.py training.use_esm=true

# 覆盖多个参数
accelerate launch --num_processes=4 main.py \
  training.use_esm=true \
  model.d_model=1024 \
  training.lr=2e-4
```

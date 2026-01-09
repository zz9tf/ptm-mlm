"""
PTM Dataset for memmap format.
Input: embeddings (pre-generated ESM embeddings) stored in memmap format
Output: original_sequence and ptm_sequence

使用 memmap 格式的优势：
1. 不需要一次性加载所有数据到内存
2. 多个进程可以共享同一个 memmap 文件（只读）
3. 数据访问更快（直接内存映射）
4. 支持延迟加载，启动速度快
"""
import os
import json
import re
import torch
import numpy as np
import random
from typing import Any, Dict, List, Optional, Tuple
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm


class PTMDatasetMemmap(TorchDataset):
    """
    PTM Dataset for memmap format.
    Input: embeddings (pre-generated ESM embeddings) stored in memmap format
    Output: original_sequence and ptm_sequence
    
    使用 memmap 格式，支持按需加载，多个进程共享同一文件。
    """

    def __init__(
        self,
        dataset_dir: str,
        device: Optional[torch.device] = None,
        seed: Optional[int] = None,
        val_size: Optional[int] = None,
        test_size: Optional[int] = None,
        preload_all: bool = False,
        use_functional_role: bool = False,
    ):
        """
        初始化 memmap 格式的 dataset。

        @param dataset_dir: 包含 memmap 文件的目录（包含 meta_mapping.json, embeddings.dat 等）
        @param device: 放置 embeddings 的设备
        @param seed: 随机种子，用于窗口选择和数据集分割
        @param val_size: 验证集样本数
        @param test_size: 测试集样本数
        @param preload_all: 是否预加载所有数据到内存（True=预加载模式，False=memmap按需加载模式）
        @param use_functional_role: 是否使用 functional role 数据（需要额外的 functional_role.dat 和 functional_role_position.dat 文件）
        """
        super().__init__()
        self.dataset_dir = dataset_dir
        self.device = device if device is not None else torch.device('cpu')
        self.rng = np.random.RandomState(seed)
        self.val_size = val_size if val_size is not None else 0
        self.test_size = test_size if test_size is not None else 0
        self.seed = seed
        self.preload_all = preload_all
        self.use_functional_role = use_functional_role
        
        # 加载元数据
        meta_mapping_path = os.path.join(self.dataset_dir, "meta_mapping.json")
        if not os.path.exists(meta_mapping_path):
            raise FileNotFoundError(f"meta_mapping.json not found in {self.dataset_dir}")
        
        with open(meta_mapping_path, 'r') as f:
            self.meta_mapping = json.load(f)
        
        self.total_samples = self.meta_mapping['total_samples']
        self.embedding_dim = self.meta_mapping['embedding_dim']
        self.sequence_length = self.meta_mapping['sequence_length']
        self.idx_to_protein_id = self.meta_mapping['idx_to_protein_id']
        self.protein_id_to_idx = self.meta_mapping['protein_id_to_idx']
        
        # 加载数据（根据 preload_all 标志选择模式）
        if self.preload_all:
            # 预加载模式：将所有数据加载到内存
            print(f"📦 Preloading all data to memory ({self.total_samples:,} samples)...")
            
            # 定义要加载的文件信息
            files_to_load = [
                {
                    "name": "embeddings.dat",
                    "dtype": np.float16,
                    "shape": (self.total_samples, self.sequence_length, self.embedding_dim),
                    "attr_memmap": "embeddings_memmap",
                    "attr_data": "embeddings_data"
                },
                {
                    "name": "orig_tokens.dat",
                    "dtype": np.int32,
                    "shape": (self.total_samples, self.sequence_length),
                    "attr_memmap": "orig_tokens_memmap",
                    "attr_data": "orig_tokens_data"
                },
                {
                    "name": "ptm_tokens.dat",
                    "dtype": np.int32,
                    "shape": (self.total_samples, self.sequence_length),
                    "attr_memmap": "ptm_tokens_memmap",
                    "attr_data": "ptm_tokens_data"
                },
                {
                    "name": "range.dat",
                    "dtype": np.int32,
                    "shape": (self.total_samples, 3),
                    "attr_memmap": "range_memmap",
                    "attr_data": "range_data"
                },
                {
                    "name": "meta_id.dat",
                    "dtype": np.int64,
                    "shape": (self.total_samples,),
                    "attr_memmap": "meta_id_memmap",
                    "attr_data": "meta_id_data"
                }
            ]

            # 如果使用functional role，添加相关文件
            if self.use_functional_role:
                files_to_load.extend([
                    {
                        "name": "functional_role.dat",
                        "dtype": np.float32,
                        "shape": (self.total_samples, self.sequence_length),
                        "attr_memmap": "functional_role_memmap",
                        "attr_data": "functional_role_data"
                    },
                    {
                        "name": "functional_role_position.dat",
                        "dtype": np.int32,
                        "shape": (self.total_samples, self.sequence_length),
                        "attr_memmap": "functional_role_position_memmap",
                        "attr_data": "functional_role_position_data"
                    }
                ])
            
            # 计算每个文件的大小并创建进度条
            total_size_bytes = 0
            file_sizes = []
            for file_info in files_to_load:
                file_size = np.prod(file_info["shape"]) * np.dtype(file_info["dtype"]).itemsize
                file_sizes.append(file_size)
                total_size_bytes += file_size
            
            total_size_gb = total_size_bytes / (1024**3)

            # 创建总体进度条（以 GB 为单位）
            pbar = tqdm(
                total=total_size_gb,
                desc="Preloading data",
                unit="GB",
                unit_scale=False,
                bar_format="{l_bar}{bar}| {n:.2f}/{total:.2f} GB"
            )
            
            # 逐个加载文件
            for idx, file_info in enumerate(files_to_load):
                file_path = os.path.join(self.dataset_dir, file_info["name"])
                file_size = file_sizes[idx]
                file_size_gb = file_size / (1024**3)

                # 更新进度条描述
                pbar.set_description(f"Loading {file_info['name']} ({file_size_gb:.2f} GB)")

                # 创建 memmap
                memmap_obj = np.memmap(
                    file_path,
                    dtype=file_info["dtype"],
                    mode='r',
                    shape=file_info["shape"]
                )
                setattr(self, file_info["attr_memmap"], memmap_obj)

                # 加载到内存（这会触发实际的数据读取）
                import time
                start_time = time.time()

                if file_info["name"] == "embeddings.dat":
                    # 🎯 embeddings.dat 使用优化块拷贝 + 低频更新策略
                    data_array = np.empty(file_info["shape"], dtype=file_info["dtype"])

                    src = memmap_obj.reshape(-1)
                    dst = data_array.reshape(-1)

                    elem_size = np.dtype(file_info["dtype"]).itemsize
                    total_elems = src.size

                    # 每次拷贝 1GB（可调到 512MB/2GB）
                    chunk_bytes = 1024 * 1024**2
                    chunk_elems = max(1, chunk_bytes // elem_size)

                    # tqdm 低频刷新：累计到 ~1GB 再 update 一次
                    update_every_gb = 1.0
                    accum_gb = 0.0

                    for i in range(0, total_elems, chunk_elems):
                        j = min(i + chunk_elems, total_elems)
                        dst[i:j] = src[i:j]

                        accum_gb += (j - i) * elem_size / (1024**3)
                        if accum_gb >= update_every_gb:
                            pbar.update(accum_gb)
                            accum_gb = 0.0

                    # 收尾
                    if accum_gb > 0:
                        pbar.update(accum_gb)
                else:
                    # 其他文件使用快速加载
                    data_array = np.array(memmap_obj)

                elapsed_time = time.time() - start_time
                load_speed_gbs = file_size_gb / elapsed_time if elapsed_time > 0 else 0

                setattr(self, file_info["attr_data"], data_array)

                # 更新进度条（更新 GB 数）- embeddings.dat 已在块拷贝时更新，这里跳过
                if file_info["name"] != "embeddings.dat":
                    pbar.update(file_size_gb)

                # 在进度条后显示文件加载信息
                pbar.write(f"  ✓ {file_info['name']}: {file_size_gb:.2f} GB loaded ({load_speed_gbs:.2f} GB/s)")

                # 立即删除 memmap 对象以节省内存
                del memmap_obj

            pbar.close()
            
            # 估算内存使用量
            embeddings_size_gb = self.embeddings_data.nbytes / (1024**3)
            tokens_size_gb = (self.orig_tokens_data.nbytes + self.ptm_tokens_data.nbytes) / (1024**3)
            range_size_gb = self.range_data.nbytes / (1024**3)
            meta_size_gb = self.meta_id_data.nbytes / (1024**3)
            total_size_gb = embeddings_size_gb + tokens_size_gb + range_size_gb + meta_size_gb
            print(f"✅ Preloaded all data to memory: {total_size_gb:.2f} GB "
                  f"(embeddings: {embeddings_size_gb:.2f} GB, tokens: {tokens_size_gb:.2f} GB, "
                  f"range: {range_size_gb:.2f} GB, meta: {meta_size_gb:.2f} GB)")
        else:
            # Memmap 模式：按需加载（多个进程可以共享）
            self.embeddings_memmap = np.memmap(
                os.path.join(self.dataset_dir, "embeddings.dat"),
                dtype=np.float16,
                mode='r',
                shape=(self.total_samples, self.sequence_length, self.embedding_dim)
            )
            self.orig_tokens_memmap = np.memmap(
                os.path.join(self.dataset_dir, "orig_tokens.dat"),
                dtype=np.int32,
                mode='r',
                shape=(self.total_samples, self.sequence_length)
            )
            self.ptm_tokens_memmap = np.memmap(
                os.path.join(self.dataset_dir, "ptm_tokens.dat"),
                dtype=np.int32,
                mode='r',
                shape=(self.total_samples, self.sequence_length)
            )
            self.range_memmap = np.memmap(
                os.path.join(self.dataset_dir, "range.dat"),
                dtype=np.int32,
                mode='r',
                shape=(self.total_samples, 3)  # [start, end, length]
            )
            self.meta_id_memmap = np.memmap(
                os.path.join(self.dataset_dir, "meta_id.dat"),
                dtype=np.int64,
                mode='r',
                shape=(self.total_samples,)
            )

            # 如果使用functional role，加载相关memmap文件
            if self.use_functional_role:
                self.functional_role_memmap = np.memmap(
                    os.path.join(self.dataset_dir, "functional_role.dat"),
                    dtype=np.float32,
                    mode='r',
                    shape=(self.total_samples, self.sequence_length)
                )
                self.functional_role_position_memmap = np.memmap(
                    os.path.join(self.dataset_dir, "functional_role_position.dat"),
                    dtype=np.int32,
                    mode='r',
                    shape=(self.total_samples, self.sequence_length)
                )
        
        # 初始化 samples_by_split
        self.samples_by_split = {'train': [], 'val': [], 'test': []}

        # 构建样本列表（根据 PTM 阈值过滤）
        self._build_samples()

        # 分割数据集
        if self.val_size > 0 or self.test_size > 0:
            self._split_samples()
        else:
            # 如果没有分割，所有样本都在 train 中
            for sample in self._all_samples:
                sample['split'] = 'train'
                self.samples_by_split['train'].append(sample)
            if hasattr(self, '_all_samples'):
                delattr(self, '_all_samples')

        # 扁平化索引以优化 __len__ 和 __getitem__ 性能
        self._build_flat_index()

    def _build_samples(self):
        """
        构建样本列表。
        
        按你的需求：不做任何 PTM 筛选/随机采样，只是「按索引顺序」把所有 sample_idx 都纳入列表，
        方便做纯 load/带宽测试。
        """
        all_samples = []
        
        print(f"🚀 Building sample list (no PTM filtering) from {self.total_samples:,} samples (memmap format)...")
        
        # 直接保留所有样本索引
        for sample_idx in range(self.total_samples):
            all_samples.append({
                "sample_idx": sample_idx,
                "split": None,
            })
        
        self._all_samples = all_samples
        print(f"✅ Built sample list: {len(all_samples):,} samples (from {self.total_samples:,} total)")
    
    def _split_samples(self):
        """
        将样本分割为 train/val/test。
        """
        all_samples = getattr(self, '_all_samples', [])
        total = len(all_samples)
        
        if self.val_size + self.test_size > total:
            raise ValueError(
                f"val_size + test_size exceeds dataset size ({self.val_size + self.test_size} > {total})"
            )
        
        # 创建索引并打乱
        indices = list(range(total))
        split_rng = random.Random(self.seed)
        split_rng.shuffle(indices)
        
        # 计算分割边界
        test_start = total - self.test_size
        val_start = test_start - self.val_size
        
        train_idx = indices[:val_start] if val_start > 0 else []
        val_idx = indices[val_start:test_start] if self.val_size > 0 else []
        test_idx = indices[test_start:] if self.test_size > 0 else []
        
        # 分配样本到各个 split（确保已初始化）
        if not hasattr(self, 'samples_by_split'):
            self.samples_by_split = {'train': [], 'val': [], 'test': []}
        
        for idx in train_idx:
            sample = all_samples[idx].copy()
            sample['split'] = 'train'
            self.samples_by_split['train'].append(sample)
        
        for idx in val_idx:
            sample = all_samples[idx].copy()
            sample['split'] = 'val'
            self.samples_by_split['val'].append(sample)
        
        for idx in test_idx:
            sample = all_samples[idx].copy()
            sample['split'] = 'test'
            self.samples_by_split['test'].append(sample)
        
        print(f"📊 Dataset split (seed={self.seed}): "
              f"Train: {len(self.samples_by_split['train'])}, "
              f"Val: {len(self.samples_by_split['val'])}, "
              f"Test: {len(self.samples_by_split['test'])}")
        
        # 清理临时存储
        if hasattr(self, '_all_samples'):
            delattr(self, '_all_samples')

    def _build_flat_index(self):
        """构建扁平化索引以优化性能"""
        # 处理完整数据集（包含所有splits）和单个split数据集的情况
        if len(self.samples_by_split) == 3 and all(k in self.samples_by_split for k in ['train', 'val', 'test']):
            # 完整数据集：拼接所有splits
            self.flat_samples = (
                self.samples_by_split["train"] +
                self.samples_by_split["val"] +
                self.samples_by_split["test"]
            )
        else:
            # 单个split数据集：直接使用该split的样本
            split_name = list(self.samples_by_split.keys())[0]
            self.flat_samples = self.samples_by_split[split_name]
        self.flat_len = len(self.flat_samples)

    def __len__(self) -> int:
        """返回数据集大小（预计算，避免每次求和）"""
        return self.flat_len
    
    def get_split_samples(self, split_name: str) -> List[Dict[str, Any]]:
        """
        获取指定 split 的样本列表。
        
        @param split_name: Split 名称 ('train', 'val', 或 'test')
        @return: 指定 split 的样本列表
        """
        return self.samples_by_split.get(split_name, [])
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        获取单个样本（使用扁平化索引以优化性能）。
        
        @param idx: 样本索引（跨所有 splits）
        @return: 包含以下字段的字典：
            - 'embeddings': torch.Tensor, shape (max_seq_len, embed_dim) float16 CPU
            - 'orig_ids': np.ndarray[int32]，原始 token ids
            - 'ptm_ids': np.ndarray[int32]，PTM token ids
            - 'seq_length': int (实际长度，用于 padding mask)
            - 'range': Tuple[int, int], (start, end) 范围
            - 'sample_idx': int, 原始样本索引
            - 'protein_idx': int, 蛋白质索引（避免字符串查找）
        """
        if idx >= self.flat_len:
            raise IndexError(f"Index {idx} out of range (total samples: {self.flat_len})")

        sample = self.flat_samples[idx]
        sample_idx = sample['sample_idx']
        
        # 根据预加载模式选择数据源
        if self.preload_all:
            # 从内存数组读取数据
            embedding = self.embeddings_data[sample_idx]  # (512, 1152) float16
            orig_tokens = self.orig_tokens_data[sample_idx]  # (512,) int32
            ptm_tokens = self.ptm_tokens_data[sample_idx]  # (512,) int32
            range_data = self.range_data[sample_idx]  # (3,) int32 [start, end, length]
            protein_idx = int(self.meta_id_data[sample_idx])

            # 如果使用functional role，读取相关数据
            if self.use_functional_role:
                functional_role = self.functional_role_data[sample_idx]  # (512,) float32
                functional_role_position = self.functional_role_position_data[sample_idx]  # (512,) int32
        else:
            # 从 memmap 中读取数据（按需加载）
            embedding = self.embeddings_memmap[sample_idx]  # (512, 1152) float16
            orig_tokens = self.orig_tokens_memmap[sample_idx]  # (512,) int32
            ptm_tokens = self.ptm_tokens_memmap[sample_idx]  # (512,) int32
            range_data = self.range_memmap[sample_idx]  # (3,) int32 [start, end, length]
            protein_idx = int(self.meta_id_memmap[sample_idx])

            # 如果使用functional role，读取相关数据
            if self.use_functional_role:
                functional_role = self.functional_role_memmap[sample_idx]  # (512,) float32
                functional_role_position = self.functional_role_position_memmap[sample_idx]  # (512,) int32
        
        # 获取蛋白质 ID
        protein_id = self.idx_to_protein_id[protein_idx]
        
        # 获取实际序列长度
        seq_length = int(range_data[2])  # length
        range_tuple = (int(range_data[0]), int(range_data[1]))  # (start, end)

        # 保持 float16 CPU tensor，避免不必要的转换和传输
        embedding_tensor = torch.from_numpy(embedding)  # 仍是 float16 CPU tensor

        result = {
            "embeddings": embedding_tensor,      # (max_seq_len, embed_dim) float16 CPU
            "orig_ids": orig_tokens,             # np.ndarray[int32]，原始 ids
            "ptm_ids": ptm_tokens,               # np.ndarray[int32]，PTM ids
            "seq_length": seq_length,
            "range": range_tuple,
            "sample_idx": sample_idx,
            "protein_idx": protein_idx,          # 直接用 int，避免字符串查找
        }

        # 如果使用functional role，添加相关数据
        if self.use_functional_role:
            result["functional_role"] = functional_role  # np.ndarray[float32]，functional role 值
            result["functional_role_position"] = functional_role_position  # np.ndarray[int32]，functional role 位置

        return result
    
    def get_split_datasets(self) -> Dict[str, Optional["PTMDatasetMemmap"]]:
        """
        获取各个 split 的数据集。
        
        @return: 包含 train/val/test PTMDatasetMemmap 的字典；val/test 可能为 None
        """
        splits = {}
        
        for split_name in ['train', 'val', 'test']:
            split_samples = self.samples_by_split[split_name]
            if not split_samples:
                splits[split_name] = None
                continue
            
            # 创建新的 dataset 实例（共享数据）
            dataset = PTMDatasetMemmap.__new__(PTMDatasetMemmap)
            dataset.dataset_dir = self.dataset_dir
            dataset.device = self.device
            dataset.rng = np.random.RandomState(self.rng.randint(0, 2**31) if self.rng is not None else None)
            dataset.preload_all = self.preload_all
            
            # 共享数据（memmap 或预加载的内存数组）
            if self.preload_all:
                # 共享预加载的内存数组
                dataset.embeddings_data = self.embeddings_data
                dataset.orig_tokens_data = self.orig_tokens_data
                dataset.ptm_tokens_data = self.ptm_tokens_data
                dataset.range_data = self.range_data
                dataset.meta_id_data = self.meta_id_data

                # 如果使用functional role，共享相关数据
                if self.use_functional_role:
                    dataset.functional_role_data = self.functional_role_data
                    dataset.functional_role_position_data = self.functional_role_position_data
            else:
                # 共享 memmap 文件（只读，可以安全共享）
                dataset.embeddings_memmap = self.embeddings_memmap
                dataset.orig_tokens_memmap = self.orig_tokens_memmap
                dataset.ptm_tokens_memmap = self.ptm_tokens_memmap
                dataset.range_memmap = self.range_memmap
                dataset.meta_id_memmap = self.meta_id_memmap

                # 如果使用functional role，共享相关memmap
                if self.use_functional_role:
                    dataset.functional_role_memmap = self.functional_role_memmap
                    dataset.functional_role_position_memmap = self.functional_role_position_memmap
            
            dataset.meta_mapping = self.meta_mapping
            dataset.total_samples = self.total_samples
            dataset.embedding_dim = self.embedding_dim
            dataset.sequence_length = self.sequence_length
            dataset.idx_to_protein_id = self.idx_to_protein_id
            dataset.protein_id_to_idx = self.protein_id_to_idx
            
            dataset.samples_by_split = {split_name: split_samples}
            dataset.seed = self.seed
            dataset.val_size = self.val_size
            dataset.test_size = self.test_size

            # 确保 split datasets 也有扁平化索引
            dataset._build_flat_index()
            
            splits[split_name] = dataset
        
        return splits
    
    def get_split_mapping(self) -> Dict[str, str]:
        """
        获取 unique_id 到 split 的映射。
        
        @return: 映射 unique_id 到 split 名称的字典
        """
        split_mapping = {}
        for split_name in ['train', 'val', 'test']:
            for sample in self.samples_by_split[split_name]:
                sample_idx = sample['sample_idx']
                if self.preload_all:
                    protein_idx = int(self.meta_id_data[sample_idx])
                else:
                    protein_idx = int(self.meta_id_memmap[sample_idx])
                protein_id = self.idx_to_protein_id[protein_idx]
                split_mapping[protein_id] = split_name
        return split_mapping


def get_ptm_dataset_memmap(
    dataset_dir: str,
    device: Optional[torch.device] = None,
    seed: Optional[int] = None,
    val_size: Optional[int] = None,
    test_size: Optional[int] = None,
    preload_all: bool = False,
    use_functional_role: bool = False,
) -> Dict[str, Optional[PTMDatasetMemmap]]:
    """
    从 memmap 格式加载 PTM 数据集并分割为 train/val/test。

    @param dataset_dir: 包含 memmap 文件的目录
    @param device: 放置 embeddings 的设备
    @param seed: 随机种子，用于窗口选择和数据集分割
    @param val_size: 验证集样本数
    @param test_size: 测试集样本数
    @param preload_all: 是否预加载所有数据到内存（True=预加载模式，False=memmap按需加载模式）
    @param use_functional_role: 是否使用 functional role 数据
    @return: 包含 train/val/test PTMDatasetMemmap splits 和 split_mapping 的字典
    """
    dataset = PTMDatasetMemmap(
        dataset_dir=dataset_dir,
        device=device,
        seed=seed,
        val_size=val_size,
        test_size=test_size,
        preload_all=preload_all,
        use_functional_role=use_functional_role,
    )
    
    # 获取 split datasets
    splits = dataset.get_split_datasets()
    
    # 获取 split mapping 并添加到返回字典
    split_mapping = dataset.get_split_mapping()
    splits["split_mapping"] = split_mapping
    
    return splits


"""
Inference pipeline for downstream tasks.
This module provides utilities to load pre-computed embeddings, process them through adaptor checkpoints,
and prepare them for head training.
"""
import os
import sys
import torch
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from tqdm import tqdm

# Add project root to sys.path for importing main_pipeline
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..', '..')
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from .inference_adaptor import AdaptorInference
from .embedding_generator_inference import EmbeddingGeneratorInference
from collections import defaultdict
import json


class InferencePipeline:
    """
    推理流水线类，用于处理从embeddings到adaptor再到head的完整流程。
    """

    def __init__(self, embeddings_base_dir: str = "/home/zz/zheng/ptm-mlm/downstream_tasks/embeddings",
                 checkpoints_base_dir: str = "/home/zz/zheng/ptm-mlm/downstream_tasks/checkpoints"):
        """
        初始化推理流水线。

        @param embeddings_base_dir: embeddings基础目录
        @param checkpoints_base_dir: checkpoints基础目录
        """
        self.embeddings_base_dir = Path(embeddings_base_dir)
        self.checkpoints_base_dir = Path(checkpoints_base_dir)

    def get_embeddings_path(self, model_name: str, layer_index: int) -> Path:
        """
        根据model_name和layer_index生成embeddings路径。

        @param model_name: 模型名称
        @param layer_index: 层索引
        @return: embeddings目录路径
        """
        # 构建embeddings目录名称
        embeddings_dir_name = f"{model_name}_layer{layer_index}"
        embeddings_path = self.embeddings_base_dir / embeddings_dir_name
        return embeddings_path

    def find_checkpoint_path(self, checkpoint_name: str) -> Path:
        """
        根据checkpoint名称找到checkpoint文件路径。

        @param checkpoint_name: checkpoint文件名（不含.ckpt扩展名）
        @return: checkpoint文件完整路径
        """
        checkpoint_path = self.checkpoints_base_dir / f"{checkpoint_name}.ckpt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return checkpoint_path

    def load_embeddings_for_task(self, model_name: str, layer_index: int, task_name: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        为特定任务加载embeddings。

        @param model_name: 模型名称
        @param layer_index: 层索引
        @param task_name: 任务名称 ('nhas', 'p_site', 'ppi')
        @return: (embeddings字典, metadata字典) 元组，embeddings字典包含训练/验证/测试embeddings和labels
        """
        embeddings_path = self.get_embeddings_path(model_name, layer_index)
        task_path = embeddings_path / task_name

        if not task_path.exists():
            raise FileNotFoundError(f"Task embeddings not found: {task_path}")

        print(f"\n📦 开始加载任务数据: {task_name}")
        print(f"   📁 任务路径: {task_path}")

        embeddings = {}
        metadata_dict = {}

        # 区分 embeddings 文件和 labels 文件
        # embeddings 文件需要 metadata，labels 文件不需要
        embedding_files = {
            'train': ['train_embeddings.pt', 'train_labels.pt'],
            'valid': ['valid_embeddings.pt', 'valid_labels.pt'],
            'test': ['test_embeddings.pt', 'test_labels.pt']
        }

        # 对于ppi任务，还有额外的embeddings
        # 🔧 PPI任务特殊处理：加载binder、wt和ptm的embeddings
        # PTM embeddings已经生成，直接load，然后通过adaptor block处理
        if task_name == 'ppi':
            embedding_files['train'].extend(['train_binder_embeddings.pt', 'train_wt_embeddings.pt', 'train_ptm_embeddings.pt'])
            embedding_files['valid'].extend(['valid_binder_embeddings.pt', 'valid_wt_embeddings.pt', 'valid_ptm_embeddings.pt'])
            embedding_files['test'].extend(['test_binder_embeddings.pt', 'test_wt_embeddings.pt', 'test_ptm_embeddings.pt'])

        # 判断是否为 labels 文件（不需要 metadata）
        def is_label_file(filename: str) -> bool:
            """判断文件是否为 labels 文件"""
            return 'labels' in filename

        # 计算总文件数（用于进度条）
        total_files = sum(len(files) for files in embedding_files.values())
        
        print(f"\n🔍 扫描文件并加载数据...")
        print(f"   预计加载 {total_files} 个文件\n")
        
        # 使用 tqdm 显示加载进度
        with tqdm(total=total_files, desc="📦 加载数据文件", unit="file", 
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
            
            for split, files in embedding_files.items():
                pbar.set_postfix({"数据集": split})
                
                for file in files:
                    file_path = task_path / file
                    pbar.set_postfix({"数据集": split, "文件": file[:25] + "..." if len(file) > 25 else file})
                    
                    if file_path.exists():
                        key = file.replace('.pt', '')
                        
                        # 加载数据文件
                        try:
                            embeddings[key] = torch.load(file_path, weights_only=False)
                            
                            # 显示加载的数据形状/大小（通过 postfix）
                            if isinstance(embeddings[key], torch.Tensor):
                                shape_str = str(list(embeddings[key].shape))
                                pbar.set_postfix({
                                    "文件": file[:20] + "..." if len(file) > 20 else file,
                                    "形状": shape_str[:30]
                                })
                            elif isinstance(embeddings[key], list):
                                pbar.set_postfix({
                                    "文件": file[:20] + "..." if len(file) > 20 else file,
                                    "长度": len(embeddings[key])
                                })
                        except Exception as e:
                            pbar.close()
                            raise RuntimeError(f"❌ 加载文件失败 {file_path}: {e}")
                        
                        # 只对 embeddings 文件（非 labels）加载 metadata
                        if not is_label_file(file):
                            metadata_file = file_path.parent / f"{file_path.stem}_metadata.json"
                            
                            if not metadata_file.exists():
                                pbar.close()
                                raise FileNotFoundError(
                                    f"❌ Metadata文件未找到: {metadata_file}\n"
                                    f"   Embeddings文件需要对应的metadata文件来处理windows和特殊token。\n"
                                    f"   请先运行inference生成embeddings和metadata。"
                                )
                            
                            try:
                                metadata_dict[key] = EmbeddingGeneratorInference.load_metadata(str(metadata_file))
                                metadata_count = len(metadata_dict[key]) if isinstance(metadata_dict[key], list) else 1
                                pbar.set_postfix({
                                    "文件": file[:20] + "..." if len(file) > 20 else file,
                                    "metadata": f"{metadata_count}条"
                                })
                            except Exception as e:
                                pbar.close()
                                raise RuntimeError(
                                    f"❌ 加载metadata失败 {metadata_file}: {e}"
                                )
                    else:
                        # 文件不存在，也更新进度条
                        pbar.set_postfix({"文件": file[:20] + "..." if len(file) > 20 else file, "状态": "⚠️不存在"})
                    
                    pbar.update(1)

        print(f"\n✅ 数据加载完成!")
        print(f"   📊 已加载 {len(embeddings)} 个数据文件")
        print(f"   📋 已加载 {len(metadata_dict)} 个metadata文件")

        # 验证：确保至少加载了训练数据
        if task_name == 'ppi':
            # 🔧 PPI 任务需要 binder、wt 和 ptm embeddings（PTM embeddings已经生成）
            required_train_keys = ['train_binder_embeddings', 'train_wt_embeddings', 'train_ptm_embeddings']
            missing_keys = [key for key in required_train_keys if key not in embeddings]
            if missing_keys:
                raise FileNotFoundError(
                    f"❌ 未找到 PPI 训练数据文件: {', '.join(missing_keys)}\n"
                    f"   任务路径: {task_path}\n"
                    f"   请先运行 embedding generation 脚本生成数据文件。\n"
                    f"   对于 PPI 任务，请运行: python utils/embeddings_generator/ppi_generator.py"
                )
        else:
            # 其他任务需要 train_embeddings
            if 'train_embeddings' not in embeddings:
                raise FileNotFoundError(
                    f"❌ 未找到训练数据文件 (train_embeddings.pt)\n"
                    f"   任务路径: {task_path}\n"
                    f"   请先运行 embedding generation 脚本生成数据文件。\n"
                    f"   对于 NHA 任务，请运行: python utils/embeddings_generator/nhas_generator.py"
                )

        return embeddings, metadata_dict

    def process_embeddings(self, embeddings: torch.Tensor,
                          metadata_list: List[Dict],
                          adaptor_checkpoint: Optional[str] = None,
                          device: str = "cuda",
                          batch_size: int = 32) -> List[torch.Tensor]:
        """
        处理 embeddings，根据是否有 checkpoint 选择处理路径。
        
        - 如果有 checkpoint：通过 adaptor 批量处理 batch embeddings，然后 merge
        - 如果没有 checkpoint：直接处理（移除特殊 token、merge windows）
        
        @param embeddings: 输入embeddings，单个大的 batch tensor (total_items, max_seq_len, embed_dim)
        @param metadata_list: metadata 列表（必需）
        @param adaptor_checkpoint: adaptor checkpoint名称。如果为 None，直接使用预训练模型的 embeddings
        @param device: 设备
        @param batch_size: Batch size for adaptor processing（如果有 checkpoint）
        @return: 处理后的embeddings列表，每个元素形状为 (seq_len, hidden_size)
        """
        if embeddings.dim() != 3:
            raise ValueError(f"Expected 3D tensor (total_items, max_seq_len, embed_dim), got {embeddings.dim()}D")

        # 验证 batch_size 参数
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        if adaptor_checkpoint is not None:
            # 有 checkpoint：通过 adaptor 批量处理
            checkpoint_path = self.find_checkpoint_path(adaptor_checkpoint)
            adaptor = AdaptorInference(str(checkpoint_path), device=device)
            
            # 🔄 按 batch_size 分批处理 embeddings
            total_items = embeddings.shape[0]
            processed_batches = []
            
            # 计算总批次数（用于进度条）
            num_batches = (total_items + batch_size - 1) // batch_size
            
            with tqdm(total=num_batches, desc="🔄 Processing batches", unit="batch") as pbar:
                for batch_idx in range(0, total_items, batch_size):
                    # 计算当前批次的结束索引
                    end_idx = min(batch_idx + batch_size, total_items)
                    
                    # 提取当前批次的 embeddings
                    batch_embeddings = embeddings[batch_idx:end_idx]
                    
                    # 🔍 找到当前批次对应的 metadata（根据 embedding_idx）
                    batch_metadata = []
                    for meta in metadata_list:
                        embedding_idx = meta.get('embedding_idx')
                        if embedding_idx is not None and batch_idx <= embedding_idx < end_idx:
                            # 创建新的 metadata 副本，更新 embedding_idx 为批次内的相对索引
                            batch_meta = meta.copy()
                            batch_meta['embedding_idx'] = embedding_idx - batch_idx
                            batch_metadata.append(batch_meta)
                    
                    # 验证：确保批次有对应的 metadata（至少应该有一个）
                    if len(batch_metadata) == 0:
                        raise ValueError(
                            f"No metadata found for batch [{batch_idx}:{end_idx}]. "
                            f"This might indicate a mismatch between embeddings and metadata."
                        )
                    
                    # 处理当前批次
                    batch_processed_tensor, _ = adaptor.process_embeddings(
                        batch_embeddings,
                        metadata_list=batch_metadata
                    )
                    
                    processed_batches.append(batch_processed_tensor)
                    
                    pbar.set_postfix({
                        "批次": f"{batch_idx // batch_size + 1}/{num_batches}",
                        "items": f"{end_idx - batch_idx}/{total_items}"
                    })
                    pbar.update(1)
            
            # 🔗 合并所有批次的处理结果
            processed_batch = torch.cat(processed_batches, dim=0)
        else:
            # 没有 checkpoint：直接使用预训练模型的 embeddings（移动到 CPU）
            processed_batch = embeddings.cpu()

        # 统一进行 merge（如果有 windows）和移除特殊 token
        final_embeddings = self._merge_embeddings(processed_batch, metadata_list)

        # 返回list格式，保持变长序列的灵活性
        return final_embeddings
    
    def _merge_embeddings(self, processed_embeddings_tensor: torch.Tensor, metadata_list: list) -> List[torch.Tensor]:
        """
        根据 metadata 合并 embeddings（如果有 windows），并移除特殊 token。
        
        @param processed_embeddings_tensor: 处理后的 batch tensor (total_items, max_seq_len, hidden_size)
        @param metadata_list: Metadata 列表
        @return: 合并后的 embeddings 列表，每个元素形状为 (seq_len, hidden_size)
        """
        # 按 sequence_id 分组
        sequence_groups = defaultdict(list)
        
        # 遍历 metadata_list，根据 embedding_idx 从大的 batch tensor 中提取对应的 embeddings
        for meta in metadata_list:
            seq_id = meta['sequence_id']
            embedding_idx = meta.get('embedding_idx')
            valid_length = meta.get('valid_length')
            
            if embedding_idx is None or embedding_idx >= processed_embeddings_tensor.shape[0]:
                raise ValueError(
                    f"Invalid embedding_idx {embedding_idx} for sequence {seq_id}. "
                    f"Total embeddings: {processed_embeddings_tensor.shape[0]}"
                )
            
            # 从大的 batch tensor 中提取对应的 embedding
            emb = processed_embeddings_tensor[embedding_idx, :valid_length, :]
            sequence_groups[seq_id].append((emb, meta))
        
        # 对每个序列进行 merge（如果需要）
        all_outputs = []
        num_sequences = max(meta['sequence_id'] for meta in metadata_list) + 1
        
        # 计算需要 merge 的序列数量（有多个 windows 的序列）
        sequences_to_merge = sum(1 for group in sequence_groups.values() if len(group) > 1)
        if sequences_to_merge > 0:
            merge_desc = f"Merging windows ({sequences_to_merge} sequences)"
        else:
            merge_desc = "Preparing final embeddings"
        
        with tqdm(total=num_sequences, desc=merge_desc, unit="seq") as pbar:
            for seq_id in range(num_sequences):
                if seq_id not in sequence_groups:
                    # 如果没有该序列的数据，抛出错误（不应该发生）
                    raise ValueError(
                        f"Missing sequence_id {seq_id} in embeddings. "
                        f"Expected {num_sequences} sequences, but sequence_id {seq_id} is missing. "
                        f"Available sequence_ids: {sorted(sequence_groups.keys())}"
                    )
                
                group = sequence_groups[seq_id]
                
                if len(group) == 1:
                    # 单个完整序列，不需要 merge
                    emb, meta = group[0]
                    # 移除特殊 token（ESM2: <cls> 和 <eos>，ESM-C: BOS 和 EOS）
                    emb = self._remove_special_tokens(emb)
                    all_outputs.append(emb)
                else:
                    # 多个 windows，需要 merge
                    windows_data = [(emb, meta['start_idx'], meta['end_idx']) for emb, meta in group]
                    merge_info_list = [meta.get('merge_info') for _, meta in group]  # 提取 merge_info
                    seq_len = group[0][1]['seq_len']  # 从 metadata 获取序列长度（不包含特殊 token）
                    
                    # 推断 hidden_size
                    hidden_size = group[0][0].shape[-1]
                    
                    merged_embeddings = self._merge_window_embeddings(
                        windows_data, seq_len, hidden_size, merge_info_list
                    )
                    all_outputs.append(merged_embeddings)
                
                pbar.update(1)
        
        return all_outputs
    
    def _merge_window_embeddings(self, windows_data: list, full_length: int, hidden_size: int, 
                                 merge_info_list: list = None):
        """
        Merge embeddings from multiple sliding windows.
        使用平均池化合并重叠区域。
        
        注意：window_emb 包含特殊 token（BOS/EOS 或 <cls>/<eos>），
        start_idx 和 end_idx 是 token 位置（相对于原始序列的 token 索引，不包含特殊 token）。
        所以需要先移除特殊 token，然后再 merge。
        
        @param windows_data: List of (embeddings_tensor, start_idx, end_idx) tuples
                            embeddings_tensor 包含特殊 token
                            start_idx 和 end_idx 是 token 位置（token 索引）
        @param full_length: Full sequence length in tokens（不包含特殊 token）
        @param hidden_size: Hidden dimension size
        @param merge_info_list: Optional list of merge_info dicts from metadata，用于记录 merge 信息
        @returns: Merged embeddings tensor of shape (full_length, hidden_size) on CPU（已移除特殊 token）
        """
        # 在 CPU 上 merge（因为 processed_embeddings 已经在 CPU 上）
        merged_embeddings = torch.zeros(full_length, hidden_size)
        count_tensor = torch.zeros(full_length)  # 记录每个位置被多少个 windows 覆盖
        
        for idx, (window_emb, start_idx, end_idx) in enumerate(windows_data):
            # window_emb 已经在 CPU 上（从 process_embeddings 返回）
            # 先移除特殊 token（BOS/EOS 或 <cls>/<eos>）
            window_emb = self._remove_special_tokens(window_emb)
            
            window_len = window_emb.shape[0]
            expected_window_len = end_idx - start_idx
            
            # 确保 window embedding 长度匹配期望的窗口长度
            if window_len != expected_window_len:
                # 如果 embedding 更长，截断到期望长度
                if window_len > expected_window_len:
                    window_emb = window_emb[:expected_window_len]
                    window_len = expected_window_len
            
            actual_end = min(start_idx + window_len, full_length)
            actual_len = actual_end - start_idx
            
            # 确保不超过 merged embeddings tensor
            if actual_end > full_length:
                actual_end = full_length
                actual_len = full_length - start_idx
            
            # 累加 embeddings 和计数
            merged_embeddings[start_idx:actual_end] += window_emb[:actual_len]
            count_tensor[start_idx:actual_end] += 1
        
        # 计算平均值：每个位置的值 = 所有覆盖该位置的 windows 的平均值
        count_tensor = torch.clamp(count_tensor, min=1.0)
        merged_embeddings = merged_embeddings / count_tensor.unsqueeze(-1)
        
        # 验证最终长度匹配期望
        final_len = merged_embeddings.shape[0]
        if final_len != full_length:
            raise RuntimeError(
                f"❌ Merged embedding length ({final_len}) != expected length ({full_length})"
            )
        
        return merged_embeddings

    def prepare_data_for_training(self, model_name: str, layer_index: int, batch_size: int,
                                task_name: str, adaptor_checkpoint: Optional[str] = None,
                                device: str = "cuda") -> Dict[str, Any]:
        """
        准备用于训练的数据，包括加载embeddings并通过adaptor处理。
        根据任务名称调用对应的任务特定处理方法。

        @param model_name: 模型名称
        @param layer_index: 层索引
        @param batch_size: Batch size for adaptor processing
        @param task_name: 任务名称 ('ppi', 'nhas', 'p_site' 等)
        @param adaptor_checkpoint: adaptor checkpoint名称。如果为 None，直接使用预训练模型的 embeddings
        @param device: 设备
        @return: 处理后的数据字典
        """
        # 加载原始embeddings和metadata
        raw_embeddings, metadata_dict = self.load_embeddings_for_task(model_name, layer_index, task_name)

        # 根据任务名称调用对应的处理方法
        if task_name == 'ppi':
            processed_data = self._prepare_ppi_data(
                raw_embeddings, metadata_dict, batch_size, adaptor_checkpoint, device
            )
        elif task_name in ['nhas', 'p_site']:
            # NHA和P-site任务使用相同的处理逻辑（序列级别的embeddings）
            processed_data = self._prepare_sequence_level_data(
                raw_embeddings, metadata_dict, batch_size, adaptor_checkpoint, device
            )
        else:
            raise ValueError(f"Unknown task_name: {task_name}. Supported tasks: 'ppi', 'nhas', 'p_site'")

        print("✅ Data preparation completed!")
        return processed_data
    
    def _process_single_embedding_type(self, raw_embeddings: Dict[str, Any], metadata_dict: Dict[str, Any],
                                     embedding_key: str, batch_size: int, adaptor_checkpoint: Optional[str],
                                     device: str, need_pooling: bool = False, return_sequence_ids: bool = False) -> Optional[List[torch.Tensor]]:
        """
        处理单个embedding类型：通过adaptor处理，merge windows，移除特殊token。
        这是所有任务共用的核心处理逻辑。

        @param raw_embeddings: 原始embeddings字典
        @param metadata_dict: Metadata字典
        @param embedding_key: Embedding的key（如 'train_binder_embeddings'）
        @param batch_size: Batch size for adaptor processing
        @param adaptor_checkpoint: Adaptor checkpoint名称
        @param device: 设备
        @param need_pooling: 是否需要池化（PPI任务需要，NHA任务不需要）
        @return: 处理后的embeddings列表，如果key不存在则返回None
        """
        if embedding_key not in raw_embeddings:
            return None
        
        metadata = metadata_dict.get(embedding_key)
        if metadata is None:
            raise ValueError(f"Metadata not found for {embedding_key}")
        
        # 使用对应的metadata进行merge，返回 List[torch.Tensor]
        # 每个元素形状为 [seq_len, hidden_size]，按sequence_id顺序排列
        sequence_embeddings = self.process_embeddings(
            raw_embeddings[embedding_key],
            metadata_list=metadata,
            batch_size=batch_size,
            adaptor_checkpoint=adaptor_checkpoint,
            device=device
        )
        
        # 如果需要池化，转换为固定大小的向量 [hidden_size]
        if need_pooling:
            sequence_embeddings = self._pool_sequence_embeddings(
                sequence_embeddings,
                pool_method='mean'
            )
        
        return sequence_embeddings
    
    def _prepare_ppi_data(self, raw_embeddings: Dict[str, Any], metadata_dict: Dict[str, Any],
                          batch_size: int, adaptor_checkpoint: Optional[str], device: str) -> Dict[str, Any]:
        """
        准备PPI任务的数据。
        PPI任务特殊处理：
        - binder和wt：直接使用load进来的原始embeddings
          * 需要merge windows（如果有多个windows）
          * 需要移除特殊token
          * 不经过adaptor block处理
        - ptm：使用load进来的PTM embeddings
          * 需要merge windows（如果有多个windows）
          * 需要移除特殊token
          * 需要经过adaptor block处理
        
        所有embeddings都需要池化为固定大小的向量。

        @param raw_embeddings: 原始embeddings字典（包含binder、wt和ptm embeddings）
        @param metadata_dict: Metadata字典
        @param batch_size: Batch size for adaptor processing
        @param adaptor_checkpoint: Adaptor checkpoint名称（仅用于PTM）
        @param device: 设备
        @return: 处理后的数据字典
        """
        processed_data = {}
        
        # 🔧 PPI任务：处理原始序列（binder和wt）- 直接使用load进来的embeddings，不经过adaptor block
        original_embedding_types = [
            'train_binder_embeddings', 'train_wt_embeddings',
            'valid_binder_embeddings', 'valid_wt_embeddings',
            'test_binder_embeddings', 'test_wt_embeddings'
        ]
        
        # 🔧 PPI任务：处理PTM embeddings - 使用load进来的embeddings，然后通过adaptor block处理
        ptm_embedding_types = [
            'train_ptm_embeddings',
            'valid_ptm_embeddings',
            'test_ptm_embeddings'
        ]
        
        # 计算总步骤数（原始序列 + PTM embeddings）
        total_steps = sum(1 for key in original_embedding_types if key in raw_embeddings)
        total_steps += sum(1 for key in ptm_embedding_types if key in raw_embeddings)
        
        with tqdm(total=total_steps, desc="Processing PPI embeddings", unit="split") as pbar:
            # 🔧 处理原始序列的embeddings（binder和wt）
            # - 需要merge windows（如果有多个windows）
            # - 需要移除特殊token
            # - 不经过adaptor block处理
            for embedding_key in original_embedding_types:
                if embedding_key not in raw_embeddings:
                    continue
                
                metadata = metadata_dict.get(embedding_key)
                if metadata is None:
                    raise ValueError(f"Metadata not found for {embedding_key}")
                
                # 处理embeddings：merge windows + 移除特殊token，但不经过adaptor block
                sequence_embeddings = self.process_embeddings(
                    raw_embeddings[embedding_key],
                    metadata_list=metadata,
                    batch_size=batch_size,
                    adaptor_checkpoint=None,  # 不经过adaptor block
                    device=device
                )
                
                # 🔧 Mean pooling: 将序列级embeddings池化为固定大小的向量
                mean_embeddings = self._pool_sequence_embeddings(
                    sequence_embeddings,
                    pool_method='mean'
                )
                
                processed_data[embedding_key] = mean_embeddings
                pbar.update(1)
            
            # 🔧 处理PTM embeddings - 使用load进来的embeddings，然后通过adaptor block处理
            for embedding_key in ptm_embedding_types:
                if embedding_key not in raw_embeddings:
                    continue
                
                metadata = metadata_dict.get(embedding_key)
                if metadata is None:
                    raise ValueError(f"Metadata not found for {embedding_key}")
                
                # 处理PTM embeddings（通过adaptor block处理）
                sequence_embeddings = self.process_embeddings(
                    raw_embeddings[embedding_key],
                    metadata_list=metadata,
                    batch_size=batch_size,
                    adaptor_checkpoint=adaptor_checkpoint,  # 经过adaptor block
                    device=device
                )
                
                # 🔧 Mean pooling: 将序列级embeddings池化为固定大小的向量
                mean_embeddings = self._pool_sequence_embeddings(
                    sequence_embeddings,
                    pool_method='mean'
                )
                
                processed_data[embedding_key] = mean_embeddings
                pbar.update(1)
            
            # 复制labels（顺序与embeddings对应）
            for split in ['train', 'valid', 'test']:
                labels_key = f'{split}_labels'
                if labels_key in raw_embeddings:
                    processed_data[labels_key] = raw_embeddings[labels_key]
         
        return processed_data
    
    def _prepare_sequence_level_data(self, raw_embeddings: Dict[str, Any], metadata_dict: Dict[str, Any],
                                    batch_size: int, adaptor_checkpoint: Optional[str], device: str) -> Dict[str, Any]:
        """
        准备序列级别任务的数据（如NHA、P-site）。
        这些任务需要保持序列级别的embeddings（不池化），用于位置级别的预测。

        @param raw_embeddings: 原始embeddings字典
        @param metadata_dict: Metadata字典
        @param batch_size: Batch size for adaptor processing
        @param adaptor_checkpoint: Adaptor checkpoint名称
        @param device: 设备
        @return: 处理后的数据字典
        """
        processed_data = {}
        
        # 序列级别任务的embedding类型列表
        embedding_types = ['train_embeddings', 'valid_embeddings', 'test_embeddings']
        
        # 计算总步骤数
        total_steps = sum(1 for key in embedding_types if key in raw_embeddings)
        
        with tqdm(total=total_steps, desc="Processing sequence-level embeddings", unit="split") as pbar:
            # 处理每种类型的embeddings
            for embedding_key in embedding_types:
                processed_embeddings = self._process_single_embedding_type(
                    raw_embeddings, metadata_dict, embedding_key,
                    batch_size, adaptor_checkpoint, device,
                    need_pooling=False
                )
                if processed_embeddings is not None:
                    processed_data[embedding_key] = processed_embeddings
                    # 复制对应的labels
                    labels_key = embedding_key.replace('_embeddings', '_labels')
                    if labels_key in raw_embeddings:
                        processed_data[labels_key] = raw_embeddings[labels_key]
                    pbar.update(1)
        
        return processed_data
    
    def _pool_sequence_embeddings(self, embeddings_list: List[torch.Tensor], pool_method: str = 'mean') -> List[torch.Tensor]:
        """
        对序列级别的embeddings进行池化，将 [seq_len, hidden_size] 转换为 [hidden_size]。
        
        @param embeddings_list: List of embeddings，每个元素形状为 [seq_len, hidden_size]，按sequence顺序排列
        @param pool_method: 池化方法，'mean' 或 'max'
        @return: List of pooled embeddings，每个元素形状为 [hidden_size]，顺序保持不变
        """
        pooled_embeddings = []
        for emb in embeddings_list:
            if emb.dim() != 2:
                raise ValueError(f"Expected 2D tensor [seq_len, hidden_size], got {emb.dim()}D tensor with shape {emb.shape}")
            
            if pool_method == 'mean':
                # Mean pooling: 对序列维度求平均
                pooled = emb.mean(dim=0)  # [hidden_size]
            elif pool_method == 'max':
                # Max pooling: 对序列维度求最大值
                pooled = emb.max(dim=0)[0]  # [hidden_size]
            else:
                raise ValueError(f"Unknown pool_method: {pool_method}. Use 'mean' or 'max'.")
            
            pooled_embeddings.append(pooled)
        
        return pooled_embeddings
    
    def _remove_special_tokens(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        移除特殊 token（ESM2: <cls> 和 <eos>，ESM-C: BOS 和 EOS）。
        在 adaptor block 处理完成后，移除特殊 token 位置。
        
        @param embeddings: Embedding tensor with shape (seq_len + 2, hidden_size) 或 (seq_len + 1, hidden_size)
        @returns: Embedding tensor with shape (seq_len, hidden_size)，已移除特殊 token
        """
        if embeddings.shape[0] > 2:
            # 移除第一个和最后一个 token（<cls>/BOS 和 <eos>/EOS）
            return embeddings[1:-1]
        elif embeddings.shape[0] == 2:
            # 只有两个 token，保留第一个（通常是 <cls>/BOS）
            return embeddings[0:1]
        else:
            # 只有一个 token，直接返回
            return embeddings
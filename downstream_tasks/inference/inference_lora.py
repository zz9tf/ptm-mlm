"""
LoRA 模型推理脚本，用于从预训练的 LoRA checkpoint 生成 block 输出。
此脚本加载 LoRA checkpoint 并仅返回 block 的输出结果（不经过 heads）。

必须使用 ESM C 600M 模型生成 embeddings。

这是一个共享模块，用于所有下游任务。
"""
import torch
from tqdm import tqdm
import sys
from pathlib import Path

# 添加主项目路径以导入模型
_main_pipeline_path = Path(__file__).parent.parent.parent / "main_pipeline"
if str(_main_pipeline_path) not in sys.path:
    sys.path.insert(0, str(_main_pipeline_path))

from models.model import PTMModel
from getters.tokenizer import PTMTokenizer


class LoRAInference:
    """
    LoRA 模型推理类，用于从预训练的 LoRA checkpoint 生成 block 输出。
    仅返回 block 的输出结果，不经过 heads。
    必须使用 ESM C 600M 模型。
    """
    
    def __init__(self, checkpoint_path: str, device: str = None, max_sequence_length: int = None):
        """
        初始化 LoRA 推理模型。
        
        @param checkpoint_path: 训练好的模型 checkpoint 路径（.ckpt 文件）
        @param device: 运行推理的设备（None 表示自动检测）
        @param max_sequence_length: tokenization 的最大序列长度。
                                   如果为 None，序列不会被截断（可能导致内存问题）。
                                   默认: 512（匹配训练配置）
        """
        # 确定设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # 加载 tokenizer
        self.tokenizer = PTMTokenizer()
        
        # 加载 LoRA checkpoint
        print(f"📦 正在从 {checkpoint_path} 加载 LoRA 模型...")
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        model_state_dict = ckpt["model"]
        model_config_dict = ckpt["config"]
        
        # 从配置中获取模型参数
        # ESM C 600M 的默认维度是 1152
        embed_dim = model_config_dict.get("embed_dim", 1152)
        vocab_size = model_config_dict.get("vocab_size", self.tokenizer.get_vocab_size())
        d_model = model_config_dict.get("d_model", 512)
        block_config = model_config_dict.get("block_config", {"type": "lora"})
        
        # 初始化 PTMModel（只使用 block，heads 不会被使用）
        self.model = PTMModel(
            embed_dim=embed_dim,
            vocab_size=vocab_size,
            d_model=d_model,
            block_config=block_config,
            heads_config=[],  # 不创建 heads，因为我们只需要 block 输出
            device=self.device,
        )
        
        # 加载模型状态
        msg = self.model.load_state_dict(model_state_dict, strict=False)
        print(f"📝 模型加载信息: {msg}")
        
        # 确保模型在正确的设备上
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 加载 ESM C 600M 模型（必须使用）
        print(f"📦 正在加载 ESM C 600M 模型...")
        try:
            # 首先检查 esm 模块是否可用
            try:
                import esm
            except ImportError:
                raise ImportError(
                    "❌ esm 模块未安装。请安装 ESM 库：\n"
                    "   pip install fair-esm\n"
                    "   或者从源码安装：\n"
                    "   git clone https://github.com/facebookresearch/esm.git\n"
                    "   cd esm && pip install -e ."
                )
            
            # 检查 esm.models 模块是否存在
            try:
                from esm.models.esmc import ESMC
            except (ImportError, AttributeError) as e:
                # 检查是否是模块结构问题
                import importlib
                esm_module = importlib.import_module('esm')
                esm_path = getattr(esm_module, '__path__', [None])[0]
                raise ImportError(
                    f"❌ 无法导入 esm.models.esmc 模块: {e}\n"
                    f"   esm 模块路径: {esm_path}\n"
                    f"   请确保安装的 esm 库版本支持 ESM C 模型。\n"
                    f"   可能需要安装特定版本：\n"
                    f"   pip install 'fair-esm>=2.0.0' 或从源码安装最新版本"
                )
            
            # 加载 ESM C 600M 模型
            self.esm_model = ESMC.from_pretrained("esmc_600m")
            self.esm_model = self.esm_model.to(self.device)
            self.esm_model.eval()
            for param in self.esm_model.parameters():
                param.requires_grad = False
            
            # ESM C 使用不同的 API
            try:
                from esm.sdk.api import ESMProtein, LogitsConfig
            except (ImportError, AttributeError) as e:
                raise ImportError(
                    f"❌ 无法导入 esm.sdk.api 模块: {e}\n"
                    f"   请确保安装的 esm 库版本支持 ESM C SDK API。"
                )
            
            self.ESMProtein = ESMProtein
            self.LogitsConfig = LogitsConfig
            self.esm_layer = 30  # 使用第30层（必须使用）
            
            print(f"✅ ESM C 600M 模型加载成功！")
            print(f"📌 使用第 {self.esm_layer} 层的输出")
        except ImportError as e:
            # ImportError 直接抛出，因为已经包含了详细的错误信息
            raise
        except Exception as e:
            raise RuntimeError(
                f"❌ 无法加载 ESM C 600M 模型: {e}\n"
                f"   错误类型: {type(e).__name__}\n"
                f"   请检查：\n"
                f"   1. esm 库是否正确安装\n"
                f"   2. esm 库版本是否支持 ESM C 600M\n"
                f"   3. 网络连接是否正常（首次使用需要下载模型）\n"
                f"   安装命令：pip install fair-esm"
            )
        
        # 从模型配置获取隐藏层大小
        self.hidden_size = d_model
        
        # 设置最大序列长度（默认 512，匹配训练配置）
        if max_sequence_length is None:
            self.max_sequence_length = getattr(self.model, 'max_sequence_length', 512)
        else:
            self.max_sequence_length = max_sequence_length
        
        print(f"✅ LoRA 模型加载成功！隐藏层大小: {self.hidden_size}")
        print(f"📏 最大序列长度: {self.max_sequence_length}")
        print(f"🔧 模式: LoRA Block (仅 block 输出，不经过 heads)")
    
    @torch.no_grad()
    def _compute_esmc_embedding(self, sequences: list):
        """
        使用 ESM C 600M 计算 embeddings。
        
        @param sequences: 蛋白质序列列表（字符串）
        @returns: ESM C embeddings 张量，形状为 (batch_size, seq_len, embed_dim)
        """
        batch_embeddings = []
        
        for seq in sequences:
            try:
                # 使用 ESM C SDK API
                protein = self.ESMProtein(sequence=seq)
                protein_tensor = self.esm_model.encode(protein)
                
                if hasattr(protein_tensor, 'error'):
                    raise RuntimeError(f"ESM C 编码失败: {protein_tensor.error}")
                
                # 获取第30层的 embeddings
                # 注意：ith_hidden_layer 参数指定层索引（从0开始，所以30层是索引29或30）
                # 根据 ESM C 文档，-1 表示最后一层，正整数表示特定层
                logits_config = self.LogitsConfig(
                    sequence=True, 
                    return_embeddings=True,
                    ith_hidden_layer=self.esm_layer  # 使用第30层
                )
                logits_output = self.esm_model.logits(protein_tensor, logits_config)
                
                # ESM C 返回的 embeddings 可能是 numpy 数组或 torch 张量
                if hasattr(logits_output, 'embeddings'):
                    embeddings = logits_output.embeddings
                    # 转换为 torch 张量（如果是 numpy 数组）
                    if not isinstance(embeddings, torch.Tensor):
                        embeddings = torch.tensor(embeddings, device=self.device)
                    else:
                        embeddings = embeddings.to(self.device)
                    embeddings = embeddings.squeeze(0)
                else:
                    raise RuntimeError("ESM C logits_output 没有 embeddings 属性")
                
                batch_embeddings.append(embeddings)
            except Exception as e:
                raise RuntimeError(f"❌ ESM C 600M 处理序列失败: {e}")
        
        # 对齐序列长度（padding）
        if len(batch_embeddings) == 0:
            raise RuntimeError("没有成功生成任何 embeddings")
        
        max_len = max(emb.shape[0] for emb in batch_embeddings)
        embed_dim = batch_embeddings[0].shape[1]
        batch_size = len(batch_embeddings)
        
        padded_embeddings = torch.zeros(batch_size, max_len, embed_dim, device=self.device)
        for i, emb in enumerate(batch_embeddings):
            seq_len = emb.shape[0]
            # 确保 emb 在正确的设备上
            if isinstance(emb, torch.Tensor):
                if emb.device != self.device:
                    emb = emb.to(self.device)
                padded_embeddings[i, :seq_len, :] = emb[:seq_len, :]
            else:
                emb_tensor = torch.tensor(emb[:seq_len], device=self.device)
                padded_embeddings[i, :seq_len, :] = emb_tensor
        
        return padded_embeddings
    
    @torch.no_grad()
    def generate_block_outputs(self, sequences: list, batch_size: int = 32, 
                                return_pooled: bool = False, max_sequence_length: int = None):
        """
        为序列列表生成 block 输出（不经过 heads）。
        
        @param sequences: 蛋白质序列列表（字符串）
        @param batch_size: 推理的批次大小
        @param return_pooled: 如果为 True，返回池化的 embeddings（平均池化）。
                            如果为 False，返回序列级别的 embeddings（所有 token）
        @param max_sequence_length: tokenization 的最大序列长度。
                                   如果为 None，使用实例的 max_sequence_length
        @returns: 如果 return_pooled=True，返回形状为 (num_sequences, hidden_size) 的张量，
                 如果 return_pooled=False，返回形状为 (num_sequences, seq_len, hidden_size) 的张量
        """
        all_outputs = []
        
        # 使用提供的 max_sequence_length 或回退到实例默认值
        max_seq_len = max_sequence_length if max_sequence_length is not None else self.max_sequence_length
        
        # 分批处理
        for i in tqdm(range(0, len(sequences), batch_size), desc="生成 block 输出"):
            batch_sequences = sequences[i:i + batch_size]
            
            # 使用 ESM C 600M 计算 embeddings
            esm_embeddings = self._compute_esmc_embedding(batch_sequences)
            
            # 对齐序列长度（如果需要截断）
            if max_seq_len is not None:
                # 截断或填充到 max_seq_len
                current_len = esm_embeddings.shape[1]
                if current_len > max_seq_len:
                    esm_embeddings = esm_embeddings[:, :max_seq_len, :]
                elif current_len < max_seq_len:
                    # 填充到 max_seq_len
                    batch_size_actual = esm_embeddings.shape[0]
                    embed_dim = esm_embeddings.shape[2]
                    padding = torch.zeros(batch_size_actual, max_seq_len - current_len, embed_dim, 
                                         device=self.device)
                    esm_embeddings = torch.cat([esm_embeddings, padding], dim=1)
            
            # 通过 block 处理 embeddings（只返回 block 输出，不经过 heads）
            block_outputs = self.model.block(esm_embeddings)  # (batch_size, seq_len, d_model)
            
            if return_pooled:
                # 对序列长度进行平均池化（排除 padding）
                # 计算实际序列长度（非零部分）
                # 注意：ESM C embeddings 可能没有明确的 padding token，我们使用非零行来判断
                seq_lengths = []
                for j, seq in enumerate(batch_sequences):
                    seq_len = len(seq)
                    if max_seq_len is not None:
                        seq_len = min(seq_len, max_seq_len)
                    seq_lengths.append(seq_len)
                
                pooled_outputs = []
                for j, seq_len in enumerate(seq_lengths):
                    seq_output = block_outputs[j, :seq_len]  # (seq_len, d_model)
                    pooled = seq_output.mean(dim=0)  # (d_model,)
                    pooled_outputs.append(pooled)
                
                batch_pooled = torch.stack(pooled_outputs, dim=0)  # (batch_size, d_model)
                all_outputs.append(batch_pooled.cpu())
            else:
                # 返回所有 token embeddings（可用于逐位置预测）
                all_outputs.append(block_outputs.cpu())
        
        # 连接所有批次
        outputs = torch.cat(all_outputs, dim=0)
        return outputs
    
    @torch.no_grad()
    def generate_per_position_block_outputs(self, sequences: list, batch_size: int = 32,
                                            max_sequence_length: int = None,
                                            use_sliding_window: bool = True,
                                            window_overlap: float = 0.5):
        """
        为序列生成逐位置的 block 输出（用于位点预测等任务）。
        对长序列使用滑动窗口以保留所有位置。
        
        @param sequences: 蛋白质序列列表（字符串）
        @param batch_size: 推理的批次大小（用于窗口处理）
        @param max_sequence_length: 单个窗口的最大序列长度。
                                   如果为 None，使用实例的 max_sequence_length
        @param use_sliding_window: 如果为 True，对长于 max_sequence_length 的序列使用滑动窗口。
                                  如果为 False，截断长序列（不推荐用于位点预测）
        @param window_overlap: 窗口之间的重叠比例（0.0 到 1.0，默认 0.5 表示 50% 重叠）。
                             更高的重叠提供更好的上下文，但需要更多计算
        @returns: 张量列表，每个形状为 (seq_len, hidden_size)
        """
        all_outputs = []
        original_lengths = [len(seq) for seq in sequences]
        
        # 使用提供的 max_sequence_length 或回退到实例默认值
        max_seq_len = max_sequence_length if max_sequence_length is not None else self.max_sequence_length
        
        # 处理每个序列
        for seq_idx, sequence in enumerate(tqdm(sequences, desc="生成逐位置 block 输出")):
            seq_len = len(sequence)
            
            # 如果序列适合一个窗口（seq_len <= max_seq_len）或禁用滑动窗口
            if max_seq_len is None or seq_len <= max_seq_len or not use_sliding_window:
                # 作为单个窗口处理
                esm_embeddings = self._compute_esmc_embedding([sequence])
                
                # 对齐到 max_seq_len（如果需要）
                if max_seq_len is not None and esm_embeddings.shape[1] > max_seq_len:
                    esm_embeddings = esm_embeddings[:, :max_seq_len, :]
                elif max_seq_len is not None and esm_embeddings.shape[1] < max_seq_len:
                    # 填充
                    current_len = esm_embeddings.shape[1]
                    embed_dim = esm_embeddings.shape[2]
                    padding = torch.zeros(1, max_seq_len - current_len, embed_dim, device=self.device)
                    esm_embeddings = torch.cat([esm_embeddings, padding], dim=1)
                
                # 通过 block 处理（需要添加 batch 维度）
                block_output = self.model.block(esm_embeddings)  # (1, seq_len, d_model)
                block_output = block_output[0]  # (seq_len, d_model)
                
                # 确保输出长度匹配原始序列长度
                output_len = block_output.shape[0]
                if output_len != seq_len:
                    if output_len < seq_len:
                        # 填充以匹配序列长度
                        pad_size = seq_len - output_len
                        padding = torch.zeros(pad_size, block_output.shape[1], device=block_output.device)
                        block_output = torch.cat([block_output, padding], dim=0)
                    else:
                        # 截断
                        block_output = block_output[:seq_len]
                
                all_outputs.append(block_output.cpu())
            else:
                # 对长序列使用滑动窗口（seq_len > max_seq_len）
                windows = self._create_sliding_windows(sequence, max_seq_len, window_overlap)
                
                # 分批处理窗口
                window_outputs_list = []
                for i in range(0, len(windows), batch_size):
                    batch_windows = windows[i:i + batch_size]
                    batch_seqs = [w[0] for w in batch_windows]
                    
                    # 计算 ESM C embeddings
                    batch_esm_embeddings = self._compute_esmc_embedding(batch_seqs)
                    
                    # 对齐到 max_seq_len
                    if batch_esm_embeddings.shape[1] > max_seq_len:
                        batch_esm_embeddings = batch_esm_embeddings[:, :max_seq_len, :]
                    
                    # 通过 block 处理每个窗口
                    batch_outputs = []
                    for j in range(batch_esm_embeddings.shape[0]):
                        window_emb = batch_esm_embeddings[j]
                        window_output = self.model.block(window_emb.unsqueeze(0))[0]  # (window_len, d_model)
                        batch_outputs.append(window_output)
                    
                    window_outputs_list.extend(batch_outputs)
                
                # 合并窗口输出
                windows_data = [(out, start, end) for out, (_, start, end) in zip(window_outputs_list, windows)]
                merged_outputs = self._merge_window_outputs(windows_data, seq_len, self.hidden_size)
                # 移动到 CPU（因为最终要返回给用户）
                all_outputs.append(merged_outputs.cpu())
        
        return all_outputs, original_lengths
    
    def _create_sliding_windows(self, sequence: str, window_size: int, overlap: float = 0.5):
        """
        为长序列创建滑动窗口。
        
        @param sequence: 输入序列字符串
        @param window_size: 每个窗口的大小
        @param overlap: 窗口之间的重叠比例（0.0 到 1.0，默认 0.5 表示 50% 重叠）
        @returns: (窗口序列, start_idx, end_idx) 元组列表
        """
        windows = []
        seq_len = len(sequence)
        step_size = max(1, int(window_size * (1 - overlap)))  # 确保 step_size >= 1
        
        start = 0
        while start < seq_len:
            end = min(start + window_size, seq_len)
            window_seq = sequence[start:end]
            windows.append((window_seq, start, end))
            
            # 移动到下一个窗口
            start += step_size
            
            # 如果还没有到达末尾，但下一个窗口会超出序列，
            # 创建一个以序列末尾结束的最终窗口
            if start < seq_len and start + window_size > seq_len:
                # 创建一个覆盖剩余部分的最终窗口
                final_start = max(0, seq_len - window_size)
                if final_start > start - step_size:  # 仅当与前一个不同时添加
                    final_window_seq = sequence[final_start:seq_len]
                    windows.append((final_window_seq, final_start, seq_len))
                break
        
        return windows
    
    def _merge_window_outputs(self, windows_data: list, full_length: int, hidden_size: int):
        """
        合并来自多个滑动窗口的输出。
        对于重叠区域，取输出的平均值。
        
        @param windows_data: (输出张量, start_idx, end_idx) 元组列表
        @param full_length: 完整序列长度
        @param hidden_size: 隐藏维度大小
        @returns: 合并的输出张量，形状为 (full_length, hidden_size)
        """
        # 确定设备（从第一个窗口输出获取）
        if len(windows_data) > 0 and len(windows_data[0]) > 0:
            device = windows_data[0][0].device if isinstance(windows_data[0][0], torch.Tensor) else self.device
        else:
            device = self.device
        
        # 初始化输出张量和计数张量用于平均（在正确的设备上）
        merged_outputs = torch.zeros(full_length, hidden_size, device=device)
        count_tensor = torch.zeros(full_length, device=device)
        
        for window_out, start_idx, end_idx in windows_data:
            # 确保 window_out 在正确的设备上
            if isinstance(window_out, torch.Tensor):
                if window_out.device != device:
                    window_out = window_out.to(device)
            else:
                window_out = torch.tensor(window_out, device=device)
            
            window_len = window_out.shape[0]
            actual_end = min(start_idx + window_len, full_length)
            actual_len = actual_end - start_idx
            
            # 将输出添加到合并张量
            merged_outputs[start_idx:actual_end] += window_out[:actual_len]
            count_tensor[start_idx:actual_end] += 1
        
        # 平均重叠区域
        # 避免除以零
        count_tensor = torch.clamp(count_tensor, min=1.0)
        merged_outputs = merged_outputs / count_tensor.unsqueeze(-1)
        
        return merged_outputs


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="从预训练 LoRA 模型生成 block 输出")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型 checkpoint 路径")
    parser.add_argument("--sequences", type=str, nargs="+", help="输入序列")
    parser.add_argument("--output", type=str, help="保存输出的路径")
    parser.add_argument("--batch_size", type=int, default=32, help="推理批次大小")
    parser.add_argument("--return_pooled", action="store_true", help="返回池化的输出")
    
    args = parser.parse_args()
    
    # 初始化推理模型
    inferencer = LoRAInference(args.checkpoint)
    
    # 生成 block 输出
    outputs = inferencer.generate_block_outputs(
        args.sequences,
        batch_size=args.batch_size,
        return_pooled=args.return_pooled
    )
    
    # 如果提供了输出路径，保存输出
    if args.output:
        torch.save(outputs, args.output)
        print(f"✅ 输出已保存到 {args.output}")
    else:
        print(f"📊 生成的输出形状: {outputs.shape}")


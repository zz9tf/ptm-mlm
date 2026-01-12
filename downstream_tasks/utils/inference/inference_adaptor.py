"""
Adaptor 推理脚本，用于接受embeddings输入并生成适配后的embeddings。
此脚本加载适配器模型（LoRA等）并对输入的embeddings进行处理。

这是一个共享模块，用于所有下游任务。
"""
import torch
from tqdm import tqdm

from main_pipeline.models.model import PTMModel

class AdaptorInference:
    """
    LoRA 模型推理类，用于从预训练的 LoRA checkpoint 生成 block 输出。
    仅返回 block 的输出结果，不经过 heads。
    必须使用 ESM C 600M 模型。
    """
    
    def __init__(self, checkpoint_path: str, device: str = None, embed_dim: int = 1152):
        """
        初始化适配器推理模型。

        @param checkpoint_path: 训练好的适配器模型 checkpoint 路径（.ckpt 文件）
        @param device: 运行推理的设备（None 表示自动检测）
        @param embed_dim: 输入embeddings的维度（默认1152，对应ESM-C 600M）
        """
        # 确定设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load adaptor checkpoint
        print(f"📦 正在从 {checkpoint_path} 加载适配器模型...")
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        model_state_dict = ckpt["model"]
        model_config_dict = ckpt["config"]

        # 从配置中获取模型参数
        embed_dim = model_config_dict.get("embed_dim", embed_dim)  # 使用参数中的embed_dim作为默认值
        vocab_size = model_config_dict.get("vocab_size", 32)  # 适配器不需要vocab
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

        # 从模型配置获取隐藏层大小
        self.hidden_size = d_model
        self.embed_dim = embed_dim

        print(f"✅ 适配器模型加载成功！输入维度: {embed_dim}, 输出维度: {d_model}")
        print(f"🔧 模式: Adaptor Block (接受embeddings输入，生成适配后输出)")
    
    
    @torch.no_grad()
    def process_embeddings(self, embeddings_list: list, return_pooled: bool = False):
        """
        处理输入的embeddings列表，通过适配器生成新的embeddings。

        @param embeddings_list: embeddings张量列表，每个形状为 (seq_len, embed_dim)
        @param return_pooled: 如果为 True，返回池化的 embeddings（平均池化）。
                            如果为 False，返回序列级别的 embeddings（所有 token）
        @returns: 如果 return_pooled=True，返回形状为 (num_sequences, hidden_size) 的张量，
                 如果 return_pooled=False，返回embeddings列表，每个形状为 (seq_len, hidden_size)
        """
        all_outputs = []

        for embeddings in tqdm(embeddings_list, desc="处理适配器 embeddings"):
            # 确保embeddings在正确的设备上
            if isinstance(embeddings, torch.Tensor):
                if embeddings.device != self.device:
                    embeddings = embeddings.to(self.device)
            else:
                embeddings = torch.tensor(embeddings, device=self.device)

            # 添加batch维度: (seq_len, embed_dim) -> (1, seq_len, embed_dim)
            if embeddings.dim() == 2:
                embeddings = embeddings.unsqueeze(0)

            # 通过适配器block处理
            adapted_embeddings = self.model.block(embeddings)  # (1, seq_len, hidden_size)

            # 移除batch维度: (1, seq_len, hidden_size) -> (seq_len, hidden_size)
            adapted_embeddings = adapted_embeddings.squeeze(0)

            if return_pooled:
                # 平均池化整个序列
                pooled = adapted_embeddings.mean(dim=0)  # (hidden_size,)
                all_outputs.append(pooled.cpu())
            else:
                # 返回逐位置embeddings
                all_outputs.append(adapted_embeddings.cpu())

        if return_pooled:
            # 返回张量: (num_sequences, hidden_size)
            outputs = torch.stack(all_outputs, dim=0)
        else:
            # 返回列表，每个元素形状为 (seq_len, hidden_size)
            outputs = all_outputs

        return outputs
    


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
    inferencer = AdaptorInference(args.checkpoint)
    
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


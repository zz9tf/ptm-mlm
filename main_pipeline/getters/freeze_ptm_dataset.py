import os
import sys
import random
import pandas as pd
import torch
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from functools import partial
from torch.utils.data import Dataset as TorchDataset, Sampler
from torch.nn.utils.rnn import pad_sequence

class PTMDataset(TorchDataset):
    """
    @description In-memory PTM dataset with optional splitting support.
    """

    def __init__(self, samples: List[Dict[str, Any]]):
        """
        @param samples Pre-tokenized samples stored fully in memory.
        """
        super().__init__()
        self.samples = samples

    def __len__(self) -> int:
        """
        @returns Dataset size.
        """
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        @param idx Index of the sample.
        @returns Sample dict containing input_ids, labels, and raw sequence.
        """
        return self.samples[idx]

    @classmethod
    def from_csv(
        cls,
        tokenizer,
        dataset_location: str,
        sequence_column_name: str,
        subsample_size: Optional[int] = None,
        max_sequence_length: Optional[int] = None,
    ) -> "PTMDataset":
        """
        @param tokenizer Tokenizer callable returning token ids.
        @param dataset_location CSV path.
        @param sequence_column_name Column containing raw sequences.
        @param subsample_size Optional sample cap.
        @param max_sequence_length Optional truncation length for tokenization.
        @returns In-memory PTMDataset instance.
        """
        df = pd.read_csv(dataset_location)
        df.drop(df.filter(regex="^Unnamed"), axis=1, inplace=True)
        df.drop_duplicates(subset=sequence_column_name, inplace=True)

        if subsample_size is not None:
            df = df.iloc[:subsample_size]

        sequences = df[sequence_column_name].tolist()
        
        # Try to get unique identifier (AC_ID or index)
        # Use AC_ID if available, otherwise use index
        if "AC_ID" in df.columns:
            unique_ids = df["AC_ID"].tolist()
        else:
            # Fallback: use index as unique identifier
            unique_ids = df.index.tolist()
        
        tokenized = tokenizer(
            sequences,
            add_special_tokens=True,
            max_sequence_length=max_sequence_length,
        )
        samples = []
        for seq, ids, uid in zip(sequences, tokenized, unique_ids):
            ids_list = list(ids)
            samples.append(
                {
                    "sequence": seq,
                    "input_ids": ids_list,
                    "labels": list(ids_list),
                    "unique_id": str(uid),  # Store unique identifier for tracking
                }
            )

        return cls(samples)

    def split(
        self,
        val_size: int,
        test_size: int,
        split_seed: Optional[int] = None,
    ) -> Dict[str, Optional["PTMDataset"]]:
        """
        @param val_size Number of samples for validation.
        @param test_size Number of samples for test.
        @param split_seed Optional seed for deterministic shuffling.
        @returns Dict with train/val/test PTMDataset; val/test may be None.
        """
        total = len(self.samples)
        if val_size + test_size > total:
            raise ValueError(
                f"val_size + test_size exceeds dataset size ({val_size + test_size} > {total})"
            )

        indices = list(range(total))
        rng = random.Random(split_seed)
        rng.shuffle(indices)

        test_start = total - test_size
        val_start = test_start - val_size

        train_idx = indices[:val_start] if val_start > 0 else []
        val_idx = indices[val_start:test_start] if val_size > 0 else []
        test_idx = indices[test_start:] if test_size > 0 else []

        def _subset(idxs: List[int], split_name: str) -> Optional["PTMDataset"]:
            if not idxs:
                return None
            subset_samples = [self.samples[i] for i in idxs]
            # Add split information to each sample for tracking
            for sample in subset_samples:
                sample["split"] = split_name
            return PTMDataset(subset_samples) if subset_samples else None

        splits = {
            "train": _subset(train_idx, "train"),
            "val": _subset(val_idx, "val"),
            "test": _subset(test_idx, "test"),
        }
        
        # Store split mapping for export
        self._split_mapping = {}
        for split_name, idxs in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
            for idx in idxs:
                unique_id = self.samples[idx].get("unique_id", str(idx))
                self._split_mapping[unique_id] = split_name
        
        return splits
    
    def get_split_mapping(self) -> Dict[str, str]:
        """
        Get mapping of unique_id to split (train/val/test).
        @returns: Dictionary mapping unique_id to split name.
        """
        if not hasattr(self, "_split_mapping"):
            return {}
        return self._split_mapping.copy()


def get_ptm_dataset(
    tokenizer,
    dataset_location: str,
    sequence_column_name: str,
    val_size: int,
    test_size: int,
    split: bool = True,
    subsample_size: Optional[int] = None,
    split_seed: Optional[int] = None,
    max_sequence_length: Optional[int] = None,
) -> Union[PTMDataset, Dict[str, Optional[PTMDataset]]]:
    """
    @param tokenizer Tokenizer instance to convert sequences to token ids.
    @param dataset_location CSV path.
    @param sequence_column_name Column containing raw sequences.
    @param val_size Validation set size.
    @param test_size Test set size.
    @param split Whether to return train/val/test splits. If False, return full PTMDataset.
    @param subsample_size Optional sample cap before tokenization.
    @param split_seed Optional seed for deterministic split.
    @param max_sequence_length Optional truncation length for tokenization.
    @returns PTMDataset or dict of PTMDataset splits. If split=True, returns dict with 'splits' and 'split_mapping' keys.
    """
    dataset = PTMDataset.from_csv(
        tokenizer=tokenizer,
        dataset_location=dataset_location,
        sequence_column_name=sequence_column_name,
        subsample_size=subsample_size,
        max_sequence_length=max_sequence_length,
    )

    if not split:
        return dataset

    splits = dataset.split(
        val_size=val_size,
        test_size=test_size,
        split_seed=split_seed,
    )
    
    # Get split mapping and add to return dict
    split_mapping = dataset.get_split_mapping()
    splits["split_mapping"] = split_mapping
    
    return splits


if __name__ == "__main__":
    """
    Quick smoke test: load three CSVs in-memory and print one sample each.
    """
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    try:
        from getters.tokenizer import PTMTokenizer
    except ImportError:
        raise ImportError("❌ 无法导入 PTMTokenizer，请确认 PYTHONPATH 配置或模块路径。")

    tokenizer = PTMTokenizer()

    datasets_to_test = [
        ("/home/zz/zheng/ptm-mlm/datasets/combined.csv", "ori_seq"),
        ("/home/zz/zheng/ptm-mlm/datasets/mamba.csv", "ori_seq"),
        ("/home/zz/zheng/ptm-mlm/datasets/ptm.csv", "ori_seq"),
    ]

    for csv_path, seq_col in datasets_to_test:
        print(f"🚀 测试加载: {csv_path}")
        if not os.path.exists(csv_path):
            print(f"❌ 文件不存在: {csv_path}")
            continue
        ds = get_ptm_dataset(
            tokenizer=tokenizer,
            dataset_location=csv_path,
            sequence_column_name=seq_col,
            val_size=0,
            test_size=0,
            split=False,
            subsample_size=4,
            split_seed=42,
            max_sequence_length=None,
        )
        print(f"✅ 数据集大小: {len(ds)}")
        try:
            sample = ds[0]
            print(f"🧪 样本0 keys: {list(sample.keys())}")
            print(f"🧪 input_ids 长度: {len(sample['input_ids'])}")
            print(f"🧪 labels 长度: {len(sample['labels'])}")
            print(f"🧪 sequence: {sample['sequence']}")
            print(f"🧪 input_ids: {sample['input_ids']}")
            print(f"🧪 labels: {sample['labels']}")
        except Exception as exc:
            print(f"⚠️ 读取样本失败: {exc}")


def crop_seq(input_ids, max_seq_len, random_crop=True):
    """
    Crop sequences to max_seq_len.
    @param input_ids: Tensor of shape (seq_len).
    @param max_seq_len: Maximum sequence length.
    @param random_crop: If True, randomly crop (for training). If False, crop from start (for consistency).
    @returns: Cropped input_ids tensor.
    """
    seq_len = len(input_ids)
    if seq_len <= max_seq_len:
        return input_ids
    else:
        if random_crop:
            # Random crop for training (data augmentation)
            start_idx = torch.randint(0, seq_len - max_seq_len + 1, (1,)).item()
        else:
            # Fixed crop from start (for consistency in embeddings)
            start_idx = 0
        return input_ids[start_idx : start_idx + max_seq_len]


class DataCollatorWithPadding:
    """
    Data collator that pads sequences and optionally batches by token count.
    """

    def __init__(
        self,
        max_tokens: int,
        tokenizer,
        batch_by_tokens: bool = False,
        max_seq_len: int = None,
        random_crop: bool = True,
    ) -> None:
        """
        @param max_tokens: Maximum tokens per batch.
        @param tokenizer: Tokenizer instance with pad_token_id.
        @param batch_by_tokens: Whether to batch by token count.
        @param max_seq_len: Optional maximum sequence length for cropping.
        @param random_crop: If True, randomly crop sequences (for training). If False, crop from start (for consistency).
        """
        self.max_tokens = max_tokens
        self.tokenizer = tokenizer
        self.batch_by_tokens = batch_by_tokens
        self.max_seq_len = max_seq_len
        self.random_crop = random_crop
        self.crop_fn = (
            partial(crop_seq, max_seq_len=max_seq_len, random_crop=random_crop)
            if max_seq_len is not None
            else lambda x: x
        )

    def __call__(self, batch):
        """
        Generate a batch of data.
        @param batch: List of dictionaries with keys 'input_ids' and 'labels'.
        @returns: Dict with 'input_ids', 'pad_mask', and 'unique_ids' (if available).
        """
        input_ids = [self.crop_fn(i["input_ids"]) for i in batch]
        # Extract unique_ids if available
        unique_ids = [i.get("unique_id", None) for i in batch]
        input_ids = pad_sequence(
            [torch.tensor(x) for x in input_ids],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        if self.batch_by_tokens:
            # Keep a few sequences to make the total number of tokens in the batch <= max_tokens
            total_tokens = input_ids.numel()
            if total_tokens > self.max_tokens:
                max_num_seq = self.max_tokens // input_ids.shape[-1] + 1
                # Randomly select max_num_seq sequences from the batch to keep
                indices = torch.randperm(len(input_ids))[:max_num_seq]
                input_ids = input_ids[indices]
                unique_ids = [unique_ids[i] for i in indices]
        pad_mask = input_ids != self.tokenizer.pad_token_id

        result = {
            "input_ids": input_ids,
            "pad_mask": pad_mask,
        }
        # Add unique_ids if available
        if any(uid is not None for uid in unique_ids):
            result["unique_ids"] = unique_ids
        return result


class SequenceLengthSampler(Sampler):
    """
    Sampler that sorts sequences by length for efficient batching.
    """

    def __init__(self, dataset, sort: bool = True, sample_len_ascending: bool = True):
        """
        @param dataset: Dataset with keys 'input_ids' and 'labels'.
        @param sort: Whether to sort by sequence length.
        @param sample_len_ascending: If True, sample shorter sequences first.
        """
        self.dataset = dataset
        self.indices = list(range(len(dataset)))
        if sort is True:
            self.indices.sort(
                key=lambda x: len(dataset[x]["input_ids"]),
                reverse=not sample_len_ascending,
            )

    def __iter__(self):
        """
        @returns: Iterator over sorted indices.
        """
        return iter(self.indices)

    def __len__(self):
        """
        @returns: Number of samples.
        """
        return len(self.indices)

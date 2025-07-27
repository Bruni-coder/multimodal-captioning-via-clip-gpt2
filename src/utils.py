import torch
import torch.nn.functional as F

def collate_fn(batch):
    """
    将 batch 中每个样本的 input_ids 按最大长度对齐（padding），
    并返回统一形状的张量。
    """
    max_len = max(len(sample["input_ids"]) for sample in batch)

    padded = [
        F.pad(sample["input_ids"], (0, max_len - len(sample["input_ids"])), value=0)
        for sample in batch
    ]

    batch_input_ids = torch.stack(padded)
    return {"input_ids": batch_input_ids}

import torch
import torch.nn as nn

class DummyModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        return self.linear(x)

def collate_fn(batch):
    max_len = max(len(sample["input_ids"]) for sample in batch)
    padded = [torch.nn.functional.pad(sample["input_ids"], (0, max_len - len(sample["input_ids"]))) for sample in batch]
    return {"input_ids": torch.stack(padded)}

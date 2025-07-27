import os
import json
import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer
from torch import nn

class MultiModalDataset(Dataset):
    def __init__(self, json_path, aligned_token_path, image_root, tokenizer: GPT2Tokenizer, max_text_len=64, model_dim=768):
        self.samples = []
        self.tokenizer = tokenizer
        self.image_root = image_root
        self.max_text_len = max_text_len
        self.model_dim = model_dim

        # 加载图文对数据
        with open(json_path, "r", encoding="utf-8") as f:
            for line in f:
                self.samples.append(json.loads(line))

        # 加载 image tokens
        with open(aligned_token_path, "r", encoding="utf-8") as f:
            self.aligned_tokens = [json.loads(line)["image_tokens"] for line in f]

        # 推断图像 token 维度
        first_token = torch.tensor(self.aligned_tokens[0], dtype=torch.float32)
        if first_token.ndim == 1:
            first_token = first_token.unsqueeze(0)
        self.image_token_dim = first_token.shape[1]

        # 创建图像投影层
        self.image_projector = nn.Linear(self.image_token_dim, self.model_dim)

        print(f"[Debug] Detected image_token_dim = {self.image_token_dim}, model_dim = {self.model_dim}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        device = self.tokenizer.model.device  # 保证 embedding/拼接设备一致

        self.image_projector = self.image_projector.to(device)

        # 图像 token 部分
        image_tokens = torch.tensor(self.aligned_tokens[idx], dtype=torch.float32)
        if image_tokens.ndim == 1:
            image_tokens = image_tokens.unsqueeze(0)  # 保证 [n_image, dim]
        projected_image_tokens = self.image_projector(image_tokens.to(device)).to(device)  # 投影 + 放GPU

        # 文本编码
        caption = self.samples[idx]["caption"]
        encoding = self.tokenizer(
            caption,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len
        )
        input_ids = encoding["input_ids"].squeeze(0).to(device)
        attention_mask = encoding["attention_mask"].squeeze(0).to(device)

        # 文本 embedding
        text_embeds = self.tokenizer.model.get_input_embeddings()(input_ids)

        # 拼接图像 + 文本 embedding
        inputs_embeds = torch.cat([projected_image_tokens, text_embeds], dim=0)  # [n_img + seq, 768]

        # 拼接 attention mask
        image_mask = torch.ones(projected_image_tokens.size(0), dtype=torch.long, device=device)
        attention_mask = torch.cat([image_mask, attention_mask], dim=0)

        # 构造 labels（图像部分为 -100）
        ignore_labels = torch.full((projected_image_tokens.size(0),), -100, dtype=torch.long, device=device)
        labels = torch.cat([ignore_labels, input_ids], dim=0)

        return {
            "inputs_embeds": inputs_embeds,         # 给 GPT2 的 embedding 输入
            "attention_mask": attention_mask,
            "labels": labels
        }

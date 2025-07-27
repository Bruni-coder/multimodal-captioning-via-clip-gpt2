import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, CLIPProcessor, CLIPModel
from torch import nn
from PIL import Image
import torch.nn.functional as F


def encode_image_to_tokens(image, device="cuda" if torch.cuda.is_available() else "cpu"):
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        vision_outputs = model.vision_model(pixel_values=inputs["pixel_values"])
        image_embedding = vision_outputs.pooler_output  # shape: [1, 768]

    return image_embedding.cpu()  # [1, 768]


class MultiModalPredictor:
    def __init__(self, model_path, image_token_dim=768, model_dim=768):
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = GPT2LMHeadModel.from_pretrained(model_path).eval()
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.device = self.model.device

        # 图像投影器：将 image_token_dim 投射到 model_dim
        self.image_projector = nn.Linear(image_token_dim, model_dim).to(self.device)

    def generate_text_from_image(self, image: Image.Image, prompt="A photo of", max_new_tokens=30):
        # 获取图像向量 & 投影到模型空间
        image_tokens = encode_image_to_tokens(image, device=self.device)  # [1, 768]
        image_embed = self.image_projector(image_tokens.to(self.device))  # [1, 768]

        # 编码提示词文本
        current_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.device)  # [1, T]
        result = prompt

        for _ in range(max_new_tokens):
            text_embed = self.model.transformer.wte(current_ids)  # [1, T, 768]

            # 拼接图像嵌入（放前面） + 文本嵌入
            inputs_embeds = torch.cat([image_embed.unsqueeze(1), text_embed], dim=1)  # [1, 1+T, 768]
            attention_mask = torch.ones(inputs_embeds.shape[:2], device=self.device)  # [1, 1+T]

            # 生成下一个 token
            with torch.no_grad():
                logits = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask).logits
                probs = F.softmax(logits[:, -1, :], dim=-1).squeeze(0)

            next_token = torch.multinomial(probs, num_samples=1)  # 采样一个 token
            decoded = self.tokenizer.decode(next_token.item())
            result += decoded
            current_ids = torch.cat([current_ids, next_token.unsqueeze(0)], dim=1)  # [1, T+1]

            if decoded.strip() in [".", self.tokenizer.eos_token]:
                break

        return result

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch import nn
import json
import torch.nn.functional as F
import os


class MultiModalPredictor:
    def __init__(self, model_path, aligned_token_path, image_token_dim=10, model_dim=768):
        # 加载 tokenizer 和 model
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2LMHeadModel.from_pretrained(model_path).eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.image_projector = nn.Linear(image_token_dim, model_dim).to(self.device)

        # 加载图像 token 数据
        with open(aligned_token_path, "r", encoding="utf-8") as f:
            self.image_token_objs = [json.loads(line) for line in f]

    def get_image_tokens_by_index(self, index):
        tokens = self.image_token_objs[index]["image_tokens"]
        return torch.tensor(tokens, dtype=torch.float32)

    def prepare_inputs(self, prompt, image_tokens):
        if image_tokens.ndim == 1:
            image_tokens = image_tokens.unsqueeze(0)
        image_tokens = self.image_projector(image_tokens.to(self.device)).unsqueeze(0)  # [1, N, 768]

        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.device)
        text_embeds = self.model.transformer.wte(input_ids).squeeze(0).unsqueeze(0)     # [1, T, 768]

        inputs_embeds = torch.cat([image_tokens, text_embeds], dim=1)                   # [1, N+T, 768]
        attention_mask = torch.ones(inputs_embeds.shape[:2], device=self.device)        # [1, N+T]
        return inputs_embeds, attention_mask, input_ids[0]

    def predict_topk(self, inputs_embeds, attention_mask, top_k=5):
        with torch.no_grad():
            outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
            logits = outputs.logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            topk = torch.topk(probs, k=top_k)
        return topk.indices[0], topk.values[0]

    def predict_next_token(self, prompt, image_index=0, top_k=5):
        image_tokens = self.get_image_tokens_by_index(image_index)
        inputs_embeds, attention_mask, _ = self.prepare_inputs(prompt, image_tokens)
        token_ids, probs = self.predict_topk(inputs_embeds, attention_mask, top_k)

        for i, (tid, prob) in enumerate(zip(token_ids, probs)):
            print(f"[{i + 1}] {repr(self.tokenizer.decode([tid.item()]))} — {prob.item():.4f}")
        return self.tokenizer.decode([token_ids[0].item()])

    def sample_next_token(self, probs, top_p=0.9):
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        cutoff_index = torch.where(cumulative_probs > top_p)[0][0] + 1
        filtered_probs = sorted_probs[:cutoff_index]
        filtered_indices = sorted_indices[:cutoff_index]
        sampled = torch.multinomial(filtered_probs, 1)
        return filtered_indices[sampled]

    def generate_text(self, prompt, image_index=0, max_new_tokens=100, top_p=0.9, stop_token="."):
        image_tokens = self.get_image_tokens_by_index(image_index)
        image_embed = self.image_projector(image_tokens.to(self.device)).unsqueeze(0)  # [1, N, 768]
        current_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.device)[0]
        result = prompt

        for _ in range(max_new_tokens):
            text_embed = self.model.transformer.wte(current_ids).squeeze(0).to(self.device)# [1, T, 768]
            inputs_embeds = torch.cat([image_embed, text_embed], dim=0).unsqueeze(0)       # [1, N+T, 768]
            attention_mask = torch.ones(inputs_embeds.shape[:2], device=self.device)

            with torch.no_grad():
                logits = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask).logits
                probs = F.softmax(logits[:, -1, :], dim=-1).squeeze(0)

            next_token = self.sample_next_token(probs, top_p=top_p)
            current_ids = torch.cat([current_ids, next_token], dim=0)
            decoded = self.tokenizer.decode(next_token.item())

            result += decoded
            if decoded.strip() in [stop_token, self.tokenizer.eos_token]:
                break

        return result

    def batch_generate(self, prompts, image_index=0, max_new_tokens=30):
        results = []
        for prompt in prompts:
            out = self.generate_text(prompt, image_index=image_index, max_new_tokens=max_new_tokens)
            results.append(out)
        return results


if __name__ == "__main__":
    predictor = MultiModalPredictor(
        model_path="../gpt2_epoch8",  # 放在项目根目录即可
        aligned_token_path="../aligned_tokens_full.jsonl",
        image_token_dim=128,
        model_dim=768
    )

    print("\nGenerate Single Prompt:")
    output = predictor.generate_text("A dog is playing", image_index=0)
    print("Result:", output)

    print("\nPredict Top-k Next Token:")
    predictor.predict_next_token("A dog is playing", image_index=0, top_k=5)

    print("\nBatch Generation:")
    batch_outputs = predictor.batch_generate(["A dog is running", "A person is cooking"], image_index=0)
    for i, text in enumerate(batch_outputs):
        print(f"Prompt {i + 1}:", text)

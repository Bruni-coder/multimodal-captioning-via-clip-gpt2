import os
import csv
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer, CLIPModel, CLIPProcessor

# ========== [Config] ==========
image_dir = "/mnt/data/Emu3-Mutimodal-NextToken/test_images"
output_csv = "../results_multi_prompt.csv"    # 输出文件
prompts = [
    "A detailed photo of",
    "This image shows",
    "A realistic image of",
    "An image of",
    "A picture that depicts"
]
max_new_tokens = 30
device = "cuda" if torch.cuda.is_available() else "cpu"

# ========== [Load Models] ==========
print("Loading models...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

gpt2 = GPT2LMHeadModel.from_pretrained("gpt2").eval().to(device)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

projector = nn.Linear(768, 768).to(device)

# ========== [Helper Functions] ==========
def encode_image(image: Image.Image) -> torch.Tensor:
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        vision_output = clip_model.vision_model(pixel_values=inputs["pixel_values"])
    return vision_output.pooler_output  # shape: [1, 768]

def generate_caption(image_embed: torch.Tensor, prompt: str) -> str:
    image_embed = projector(image_embed).unsqueeze(1)  # [1, 1, 768]
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    result = prompt

    for _ in range(max_new_tokens):
        text_embed = gpt2.transformer.wte(input_ids)
        inputs_embeds = torch.cat([image_embed, text_embed], dim=1)
        attn_mask = torch.ones(inputs_embeds.shape[:2], device=device)

        with torch.no_grad():
            logits = gpt2(inputs_embeds=inputs_embeds, attention_mask=attn_mask).logits
            probs = F.softmax(logits[:, -1, :], dim=-1).squeeze(0)

        next_token = torch.multinomial(probs, num_samples=1)
        decoded = tokenizer.decode(next_token.item())
        result += decoded
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

        if decoded.strip() in [".", tokenizer.eos_token]:
            break

    return result

def compute_similarity(image: Image.Image, text: str) -> float:
    inputs = clip_processor(text=[text], images=image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        logits = clip_model(**inputs).logits_per_image
    return logits.item()

# ========== [Run Inference] ==========
results = []

for filename in os.listdir(image_dir):
    if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    path = os.path.join(image_dir, filename)
    try:
        image = Image.open(path).convert("RGB")
        print(f"Processing: {filename}")
        image_vec = encode_image(image)

        for prompt in prompts:
            caption = generate_caption(image_vec, prompt)
            score = compute_similarity(image, caption)
            token_count = len(tokenizer(caption).input_ids)
            results.append((filename, prompt, caption, token_count, round(score, 4)))

    except Exception as e:
        print(f"Error processing {filename}: {e}")

# ========== [Save to CSV] ==========
with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Image", "Prompt", "Caption", "Token Count", "CLIP Score"])
    writer.writerows(results)

print(f"Done! Results saved to {output_csv}")
